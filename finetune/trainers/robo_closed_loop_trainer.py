from PIL import Image
from typing import Any, Dict, List, Tuple, Optional
import os
import torch
from diffusers import (
    CogVideoXDDIMScheduler
)
from transformers import AutoTokenizer, T5EncoderModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from typing_extensions import override
from accelerate import Accelerator
from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from finetune.pipeline.pipeline_robo_closed_loop_i2va import RoboClosedLoopTransformerPipeline
from finetune.transformer.robo_transformer import RoboTransformer
from finetune.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX


class RoboClosedLoopTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"Initialized {self.__class__.__name__}")
        self.action_loss_weight = kwargs.get("action_loss_weight", 1.0)
        self.video_loss_weight = kwargs.get("video_loss_weight", 1.0)
        self.pipeline = None  # Will be initialized later

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = self.args.model_path
        
        # Set the pipeline class
        components.pipeline_cls = RoboClosedLoopTransformerPipeline

        # Load the text encoder
        tokenizer_path = os.path.join(model_path, "tokenizer")
        print(f"Loading tokenizer from {tokenizer_path}")
        components.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True  # Use this to ensure we load from local path
        )

        text_encoder_path = os.path.join(model_path, "text_encoder")
        print(f"Loading text encoder from {text_encoder_path}")
        components.text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_path,
            local_files_only=True  # Use this to ensure we load from local path
        )

        # Load the VAE
        vae_path = os.path.join(model_path, "vae")
        print(f"Loading VAE from {vae_path}")
        components.vae = AutoencoderKLCogVideoX.from_pretrained(
            vae_path,
            ignore_mismatched_sizes=True,  # 忽略大小不匹配的参数
            low_cpu_mem_usage=False,  # 如果内存足够，可以设为False以允许完全加载到内存
            local_files_only=True  # Use this to ensure we load from local path
        )

        # Load the transformer model
        transformer_path = os.path.join(model_path, "transformer")
        print(f"Loading transformer from {transformer_path}")
        components.transformer = self.get_transformer(transformer_path)

        # Load the scheduler
        scheduler_path = os.path.join(model_path, "scheduler")
        print(f"Loading scheduler from {scheduler_path}")
        components.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            scheduler_path,
            local_files_only=True  # Use this to ensure we load from local path
        )

        return components

    @override
    def initialize_pipeline(self) -> RoboClosedLoopTransformerPipeline:
        vae = self.components.vae
        
        # Enable memory optimization techniques
        if self.args.enable_tiling:
            print("[INFO] Enabling VAE tiling for memory optimization")
            vae.enable_tiling()
        
        if self.args.enable_slicing:
            print("[INFO] Enabling VAE slicing for memory optimization")
            vae.enable_slicing()
        
        # Enable gradient checkpointing for VAE if needed
        if self.args.gradient_checkpointing:
            print("[INFO] Enabling gradient checkpointing for memory optimization")
            if hasattr(vae, 'enable_gradient_checkpointing'):
                vae.enable_gradient_checkpointing()
            
        # Create the pipeline
        pipeline = RoboClosedLoopTransformerPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=vae,
            transformer=self.components.transformer,
            scheduler=self.components.scheduler,
        )
        
        return pipeline

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding
    
    @override
    def encode_action(self, action_tensor: torch.Tensor) -> torch.Tensor:
        # action_tensor shape: [T, 8], already normalized
        return action_tensor.to(self.accelerator.device)

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "action_embeddings": []}

        for sample in samples:
            ret["encoded_videos"].append(sample["encoded_video"])
            ret["prompt_embedding"].append(sample["prompt_embedding"])
            ret["images"].append(sample["image"])
            ret["action_embeddings"].append(sample["action_embedding"])

        for key in ret:
            ret[key] = torch.stack(ret[key])

        return ret
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # 1) Read batch data
        prompt_embedding = batch["prompt_embedding"]  # [B, seq_len, hidden_size]
        latent = batch["encoded_videos"]             # [B, C, F, H, W]
        images = batch["images"]                     # [B, C, H, W]
        action_gt = batch["action_embeddings"]       # [B, T, D], D=8
        
        # Get past frames and actions if available in the batch
        past_frames_actions = None
        if "past_frames" in batch and "past_actions" in batch:
            past_frames = batch["past_frames"]  # [B, C, F, H, W]
            past_actions = batch["past_actions"]  # [B, T, D]
            past_frames_actions = (past_frames, past_actions)

        # Handle patch size temporal requirements
        patch_size_t = getattr(self.state.transformer_config, "patch_size_t", None)
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            if ncopy != 0:
                first_frame = latent[:, :, :1, :, :]  # [B, C, 1, H, W]
                latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0, \
                f"[ERROR] num_frames={latent.shape[2]} not divisible by patch_size_t={patch_size_t}"

        batch_size, num_channels, num_frames, height, width = latent.shape
        prompt_embedding = prompt_embedding.to(dtype=latent.dtype)

        # 2) Encode image
        images = images.unsqueeze(2)  # [B, C, 1, H, W]
        image_latents = self.components.vae.encode(images.to(dtype=self.components.vae.dtype)).latent_dist.sample()
        image_latents = image_latents * self.components.vae.config.scaling_factor

        # 3) Sample random timesteps
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        ).long()

        # 4) Add noise to the video latents
        noise = torch.randn_like(latent)
        noisy_latents = self.components.scheduler.add_noise(latent, noise, timesteps)
        
        # 5) Prepare input for transformer
        latent_permuted = latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        noisy_latents_permuted = noisy_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        image_latents_permuted = image_latents.permute(0, 2, 1, 3, 4)  # [B, 1, C, H, W]

        # 确保 pipeline 已经初始化
        if not hasattr(self, "pipeline") or self.pipeline is None:
            self.pipeline = self.initialize_pipeline()
            print(f"[INFO] Pipeline initialized during compute_loss")
        
        # Use the pipeline's train_step method to get loss values
        try:
            loss_dict = self.pipeline.train_step(
                image=images,
                prompt=None,  # We're using prompt_embedding directly
                target_video=latent_permuted,
                target_action=action_gt,
                noisy_latents=noisy_latents_permuted,
                noise=noise,
                timesteps=timesteps,
                prompt_embeds=prompt_embedding,
                past_frames_actions=past_frames_actions,
                device=self.accelerator.device,
            )
        except Exception as e:
            print(f"[ERROR] Error during pipeline.train_step: {str(e)}")
            # 尝试重新初始化 pipeline
            self.pipeline = self.initialize_pipeline()
            print(f"[INFO] Pipeline re-initialized after error")
            
            # 再次尝试
            loss_dict = self.pipeline.train_step(
                image=images,
                prompt=None,
                target_video=latent_permuted,
                target_action=action_gt,
                noisy_latents=noisy_latents_permuted,
                noise=noise,
                timesteps=timesteps,
                prompt_embeds=prompt_embedding,
                past_frames_actions=past_frames_actions,
                device=self.accelerator.device,
            )
        
        # Apply loss weights
        weighted_video_loss = loss_dict['video_loss'] * self.video_loss_weight
        weighted_action_loss = loss_dict['action_loss'] * self.action_loss_weight
        
        # Combined loss
        total_loss = weighted_video_loss + weighted_action_loss

        # Log losses
        self.accelerator.log({
            "video_loss": loss_dict['video_loss'].item(),
            "action_loss": loss_dict['action_loss'].item(),
            "total_loss": total_loss.item(),
        }, step=self.state.global_step)

        return total_loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: RoboClosedLoopTransformerPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt, image = eval_data["prompt"], eval_data["image"]

        # Use feedback from previous steps if available
        past_frames_actions = None
        if "past_frames_actions" in eval_data:
            past_frames_actions = eval_data["past_frames_actions"]

        outputs = pipe(
            image=image,
            prompt=prompt,
            num_frames=self.state.train_frames,
            num_inference_steps=30,  # Reduced for faster validation
            task_type='joint',  # Generate both video and actions
            past_frames_actions=past_frames_actions,
        )
        
        # Return both video frames and visualized actions if available
        result = [("video", outputs.frames)]
        
        if outputs.actions is not None:
            # Visualize actions (this is a placeholder - implement visualization as needed)
            action_viz = self.visualize_actions(outputs.actions)
            result.append(("actions", action_viz))
            
        return result

    def visualize_actions(self, actions):
        """
        Convert action tensors to a visualization
        This is a placeholder - implement visualization based on your action format
        """
        # Simple visualization as a heatmap
        import numpy as np
        from PIL import Image
        
        # Convert actions to numpy and create a heatmap
        action_array = actions.detach().cpu().numpy()
        
        # Normalize to 0-255 range for visualization
        action_array = (action_array - action_array.min()) / (action_array.max() - action_array.min() + 1e-8) * 255
        action_array = action_array.astype(np.uint8)
        
        # Create a simple heatmap image
        # This is very simplistic - replace with better visualization if needed
        if len(action_array.shape) == 2:  # [T, D]
            # Create a simple colored heatmap
            heatmap = np.zeros((action_array.shape[0], action_array.shape[1], 3), dtype=np.uint8)
            heatmap[:, :, 0] = action_array  # Red channel
            return Image.fromarray(heatmap)
        else:  # [B, T, D]
            # Just visualize the first item in batch
            heatmap = np.zeros((action_array.shape[1], action_array.shape[2], 3), dtype=np.uint8)
            heatmap[:, :, 0] = action_array[0]  # Red channel
            return Image.fromarray(heatmap)

    def inference(self, image: torch.Tensor, prompt: str, task_type: str = 'joint', past_frames_actions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Run inference with the model
        
        Args:
            image: Input conditioning image
            prompt: Text prompt
            task_type: Type of task to perform - 'video', 'action', or 'joint'
            past_frames_actions: Optional tuple of (past_frames, past_actions) for feedback
            
        Returns:
            Dictionary containing generated frames and/or actions
        """
        # Create pipeline if not already created
        if not hasattr(self, "pipeline") or self.pipeline is None:
            self.pipeline = self.initialize_pipeline()
            
        # Run inference
        outputs = self.pipeline(
            image=image,
            prompt=prompt,
            num_frames=self.state.train_frames,
            num_inference_steps=50,
            task_type=task_type,
            past_frames_actions=past_frames_actions,
        )
        
        result = {}
        if task_type in ['video', 'joint']:
            result['frames'] = outputs.frames
        if task_type in ['action', 'joint']:
            result['actions'] = outputs.actions
        
        # Save past frames and actions for future use
        if hasattr(outputs, 'past_frames_actions') and outputs.past_frames_actions is not None:
            result['past_frames_actions'] = outputs.past_frames_actions
            
        return result
    
    def save_model(self, output_dir: str, global_step: int):
        # Create output directory for the transformer
        transformer_step_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(transformer_step_dir, exist_ok=True)
        print(f"[\u2713] Saving transformer model to {transformer_step_dir}")

        # Save the unified transformer model (which includes the action predictor)
        if self.components.transformer is not None:
            print("[✓] Saving RoboTransformer...")
            unwrapped_transformer = unwrap_model(self.accelerator, self.components.transformer)
            unwrapped_transformer.save_pretrained(transformer_step_dir)
        else:
            print("[⚠️] No transformer found. Skipping save.")

        print(f"[\u2713] Models saved to {transformer_step_dir}")

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

    def setup(self, state, accelerator):
        """Setup trainer with state and accelerator"""
        super().setup(state, accelerator)
        
        # Initialize pipeline right after setup is completed
        try:
            print(f"[INFO] Initializing pipeline during setup...")
            self.pipeline = self.initialize_pipeline()
            print(f"[INFO] Pipeline successfully initialized in setup()")
            
            # 打印关键组件，确认初始化正常
            if self.pipeline is not None:
                print(f"[INFO] Pipeline components check: tokenizer={self.pipeline.tokenizer is not None}, "
                      f"text_encoder={self.pipeline.text_encoder is not None}, "
                      f"vae={self.pipeline.vae is not None}, "
                      f"transformer={self.pipeline.transformer is not None}")
            else:
                print(f"[WARNING] Pipeline is None after initialization")
        except Exception as e:
            print(f"[ERROR] Failed to initialize pipeline during setup: {str(e)}")
            print(f"[INFO] Will attempt to initialize pipeline during compute_loss")
        
        # Log model initialization
        print(f"[✓] RoboTransformer initialized with:")
        print(f"    - Video loss weight: {self.video_loss_weight}")
        print(f"    - Action loss weight: {self.action_loss_weight}")
        
        action_predictor = self.components.transformer.action_predictor
        if action_predictor:
            print(f"    - Action predictor: {action_predictor.__class__.__name__}")
        else:
            print(f"    - Action predictor: None (this is a problem!)")

    def demo_feedback(self, image: torch.Tensor, prompt: str, num_steps: int = 5) -> Dict[str, Any]:
        """
        演示如何使用反馈机制生成连续的视频和动作序列
        
        Args:
            image: 初始输入图像
            prompt: 文本提示
            num_steps: 要生成的连续步骤数
            
        Returns:
            包含所有生成帧和动作的字典
        """
        # 确保pipeline已创建
        if not hasattr(self, "pipeline") or self.pipeline is None:
            self.pipeline = self.initialize_pipeline()
            
        all_frames = []
        all_actions = []
        past_frames_actions = None
        
        print(f"开始生成连续视频和动作序列，共{num_steps}步...")
        
        # 逐步生成视频和动作，每一步使用前一步的输出作为反馈
        for step in range(num_steps):
            print(f"生成第 {step+1}/{num_steps} 步...")
            
            # 运行推理，将前一步的输出作为反馈输入
            outputs = self.pipeline(
                image=image,
                prompt=prompt,
                num_frames=self.state.train_frames,
                num_inference_steps=50,
                task_type='joint',
                past_frames_actions=past_frames_actions,
            )
            
            # 将生成的帧和动作添加到结果列表中
            all_frames.append(outputs.frames)
            all_actions.append(outputs.actions)
            
            # 更新反馈信息，用于下一步生成
            past_frames_actions = outputs.past_frames_actions
            
            # 可选：使用最后一帧作为下一步的输入图像
            if step < num_steps - 1:
                # 从生成的视频中提取最后一帧作为下一步的输入
                last_frame = outputs.frames[-1].unsqueeze(0)  # 添加批次维度
                image = last_frame
                
                # 更新提示，可以根据生成的动作或结果调整
                # prompt = f"{prompt} + 步骤 {step+1} 的反馈"
        
        print(f"连续生成完成，共生成 {num_steps} 个步骤的视频和动作")
        
        # 返回结果
        return {
            "all_frames": all_frames,  # 每个元素是一个视频的帧序列
            "all_actions": all_actions,  # 每个元素是一组动作
        }

    def fit(self):
        """执行完整的训练流程"""
        print(f"[INFO] Starting fit method in RoboClosedLoopTrainer")
        
        # 在开始训练前，确保 pipeline 已初始化
        if not hasattr(self, "pipeline") or self.pipeline is None:
            print(f"[INFO] Pipeline not initialized yet, initializing now...")
            try:
                self.pipeline = self.initialize_pipeline()
                print(f"[INFO] Pipeline successfully initialized in fit()")
            except Exception as e:
                print(f"[ERROR] Failed to initialize pipeline in fit(): {str(e)}")
        
        # 调用父类的 fit 方法开始训练
        print(f"[INFO] Calling parent fit method")
        super().fit()

    def get_transformer(self, transformer_path: str) -> RoboTransformer:
        """
        Helper function to load the RoboTransformer model,
        handling the ActionPredictor component properly
        """
        if os.path.exists(transformer_path):
            print(f"[✓] Loading RoboTransformer from {transformer_path}")
            
            # Use the custom from_pretrained method that handles ActionPredictor properly
            transformer = RoboTransformer.from_pretrained(
                transformer_path,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False,
                local_files_only=True  # Use this to ensure we load from local path
            )
            
            # Check if ActionPredictor was properly initialized
            if hasattr(transformer, 'action_predictor'):
                print(f"[✓] ActionPredictor loaded/initialized successfully")
            else:
                print(f"[⚠️] ActionPredictor not found in the loaded model")
        else:
            print(f"[⚠️] No pretrained RoboTransformer found. Initializing new model.")
            # Initialize with default parameters - these should be adjusted based on your specific needs
            transformer = RoboTransformer()
            print(f"[✓] New RoboTransformer initialized with ActionPredictor")
            
        return transformer
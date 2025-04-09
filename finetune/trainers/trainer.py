from PIL import Image
from typing import Any, Dict, List, Tuple
import os
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoActionPipeline,
    CogVideoXTransformer3DModel,
)
from transformers import AutoTokenizer, T5EncoderModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from typing_extensions import override
from accelerate import Accelerator
from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from pipeline.pipeline_robo_transformer_i2va import RoboTransformerPipeline


class CogVideoXImageToVideoActionTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"Initialized {self.__class__.__name__}")

        self.action_predictor = self.components.action_predictor

    @override
    def load_components(self) -> Dict[str, Any]:
        from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_action import (
            CogVideoXImageToVideoActionPipeline,
            ActionPredictor,
        )

        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoActionPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        components.text_encoder = T5EncoderModel.from_pretrained(os.path.join(model_path, "text_encoder"))
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(os.path.join(model_path, "transformer"))
        components.vae = AutoencoderKLCogVideoX.from_pretrained(os.path.join(model_path, "vae"))
        components.scheduler = CogVideoXDDIMScheduler.from_pretrained(os.path.join(model_path, "scheduler"))

        # Try loading pretrained ActionPredictor
        action_predictor_path = os.path.join(model_path, "action_predictor")
        config_path = os.path.join(action_predictor_path, "config.json")

        if os.path.exists(config_path):
            print(f"[✓] Loading action_predictor from {action_predictor_path}")
            components.action_predictor = ActionPredictor.from_pretrained(action_predictor_path)
        else:
            print(f"[⚠️] No pretrained action_predictor found. Let pipeline initialize.")
            components.action_predictor = None

        # 不挂到 components，只是临时创建来拿 fallback 的 action_predictor
        pipeline = components.pipeline_cls(
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            vae=components.vae,
            transformer=components.transformer,
            scheduler=components.scheduler,
            action_predictor=components.action_predictor,
        )

        # fallback 初始化（从 pipeline 取回 predictor）
        components.action_predictor = pipeline.action_predictor
        print(f"[DEBUG] Final action_predictor: {components.action_predictor.__class__.__name__}")

        return components


    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoActionPipeline:
        pipe = CogVideoXImageToVideoActionPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
            action_predictor=self.components.action_predictor,
        )
        return pipe

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
        # 1) 读取 batch 数据
        prompt_embedding = batch["prompt_embedding"]  # [B, seq_len, hidden_size]
        latent = batch["encoded_videos"]             # [B, C, F, H, W]
        images = batch["images"]                     # [B, C, H, W]
        action_gt = batch["action_embeddings"]       # [B, T, D], D=8

        # Get the number of action frames (e.g., 41)
        num_action_frames = action_gt.shape[1]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            if ncopy != 0:
                first_frame = latent[:, :, :1, :, :]  # [B, C, 1, H, W]
                latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0, \
                f"[ERROR] num_frames={latent.shape[2]} not divisible by patch_size_t={patch_size_t}"


        batch_size, num_channels, num_frames, height, width = latent.shape
        prompt_embedding = prompt_embedding.to(dtype=latent.dtype)

        # 2) 图像编码
        images = images.unsqueeze(2)  # [B, C, 1, H, W]
        image_noise_sigma = torch.exp(
            torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        ).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latents = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist.sample()
        image_latents = image_latents * self.components.vae.config.scaling_factor

        # 3) timestep 随机采样
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        ).long()

        # 4) 加噪并输入 Transformer
        latent = latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, 1, C, H, W]

        # 扩展 image_latents 为每帧图像
        latent_padding = image_latents.new_zeros((latent.shape[0], latent.shape[1] - 1, *latent.shape[2:]))
        image_latents = torch.cat([image_latents, latent_padding], dim=1)  # [B, F, C, H, W]

        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)  # [B, F, 2C, H, W]

        # Rotary embedding 准备
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        ofs_emb = (
            None if transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )

        # Transformer预测视频噪声
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        # 5) video loss
        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        video_loss = torch.mean(
            (weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1
        ).mean()

        # 6) action loss（不使用 video_latents，仅 prompt + image）
        prompt_pooled = prompt_embedding.mean(dim=1)               # [B, D]
        pooled_image_latent = image_latents[:, 0].mean(dim=[2, 3]) # [B, C]

        # Adjust action input dimensions to match the number of action frames
        action_input_context = torch.cat([prompt_pooled, pooled_image_latent], dim=-1)  # [B, D+C]
        action_input = action_input_context.unsqueeze(1).repeat(1, num_action_frames, 1)  # [B, T, D+C]

        assert action_input.shape[-1] == self.action_predictor.latent_dim, \
            f"[ERROR] ActionPredictor expects {self.action_predictor.latent_dim}, got {action_input.shape[-1]}"
        
        # Generate time steps for action prediction
        timesteps = torch.arange(num_action_frames, device=action_gt.device).unsqueeze(0).expand(action_gt.shape[0], -1)

        action_pred_dict = self.action_predictor(action_input, timesteps)
        pred_rotation = action_pred_dict["rotation_delta"]
        pred_gripper = action_pred_dict["open_gripper"].float()
        pred_vector = action_pred_dict["world_vector"]
        pred_terminate = action_pred_dict["terminate_episode"]

        action_pred = torch.cat([pred_rotation, pred_gripper, pred_vector, pred_terminate], dim=-1)  # [B, T, 8]
        assert action_pred.shape == action_gt.shape, \
            f"[ERROR] Shape mismatch: pred {action_pred.shape}, gt {action_gt.shape}"

        action_loss = torch.nn.functional.mse_loss(action_pred, action_gt)

        total_loss = video_loss + action_loss

        # 检查梯度是否计算成功
        for name, param in self.action_predictor.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"[GRAD] {name:<30} grad norm = {param.grad.norm().item():.6f}")
            elif param.requires_grad and param.grad is None:
                print(f"[NO GRAD] {name:<30} grad is None!")

        return total_loss
    

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoActionPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def inference(self, image: torch.Tensor, prompt: str) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        # Prepare inputs
        prompt_embedding = self.encode_text(prompt)
        encoded_image = self.encode_video(image.unsqueeze(0))

        # Run the pipeline
        pipeline_output = self.components.pipeline_cls(
            prompt_embedding=prompt_embedding,
            encoded_image=encoded_image,
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            generator=self.state.generator,
        )

        return pipeline_output.frames, pipeline_output.actions
    
    
    def save_model(self, output_dir: str, global_step: int):
        # 创建存储 ActionPredictor 的文件夹
        action_predictor_dir = "/disk0/home/kuowei/ActionPredictor"
        os.makedirs(action_predictor_dir, exist_ok=True)
        print(f"[\u2713] Saving ActionPredictor model to {action_predictor_dir}")

        # 仅保存 ActionPredictor 权重
        if self.components.action_predictor is not None:
            action_predictor_step_dir = os.path.join(action_predictor_dir, f"checkpoint_step_{global_step}")
            print("[✓] Saving ActionPredictor...")
            self.components.action_predictor.save_pretrained(action_predictor_step_dir)
        else:
            print("[⚠️] No ActionPredictor found. Skipping save.")

        print(f"[\u2713] ActionPredictor saved to {action_predictor_step_dir}")

        # 为 transformer 创建输出目录
        transformer_step_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(transformer_step_dir, exist_ok=True)
        print(f"[\u2713] Saving transformer model to {transformer_step_dir}")

        # 保存 transformer 权重
        if self.components.transformer is not None:
            print("[✓] Saving transformer...")
            self.components.transformer.save_pretrained(transformer_step_dir)
        else:
            print("[⚠️] No transformer found. Skipping save.")

        print(f"[\u2713] Transformer saved to {transformer_step_dir}")

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
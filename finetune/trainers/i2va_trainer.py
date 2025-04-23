import os
import math
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    CogVideoXDDIMScheduler, 
    CogVideoXDPMScheduler
)
from transformers import T5EncoderModel, T5Tokenizer

# from finetune.constants import MODEL_PATH
from finetune.schemas import Components
from finetune.pipeline.pipeline_rvaf_i2va import RoboVideoActionFusionPipeline
from finetune.transformer.rvaf_transformer import RoboTransformer
from finetune.trainer import Trainer


class I2VATrainer(Trainer):
    """
    Trainer for Image-to-Video-to-Action (I2VA) model.
    
    This trainer supports:
    1. Training with action data alongside image, prompt, and video
    2. Generating both video and actions during inference
    
    The model takes an image and prompt as input and produces both a video and
    predicted robot actions as output.
    """
    
    # Components to unload during training to save memory
    UNLOAD_LIST = ["vae", "text_encoder"]

    def load_components(self) -> Components:
        """
        Load model components required for I2VA training.
        """
        model_path = self.args.model_path or os.environ.get(MODEL_PATH)
        if model_path is None:
            raise ValueError("No model path specified. Please set MODEL_PATH environment variable or provide model_path argument.")
        
        # Load tokenizer and text encoder for processing prompts
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        
        # Load VAE for encoding/decoding images and videos
        from diffusers.models import AutoencoderKLCogVideoX
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        
        # Load transformer for the diffusion model with action support
        transformer = RoboTransformer.from_pretrained(
            model_path, 
            subfolder="transformer",
            action_dim=self.args.action_dim
        )
        
        # Create action predictor component (if needed as a separate component)
        action_predictor = None  # This is included in the transformer model
        
        # Load scheduler
        scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        # Pipeline class used for inference
        pipeline_cls = RoboVideoActionFusionPipeline
        
        # Return all components
        return Components(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            action_predictor=action_predictor,
            scheduler=scheduler,
            pipeline_cls=pipeline_cls,
        )
    
    def initialize_pipeline(self) -> DiffusionPipeline:
        """
        Initialize a pipeline for inference.
        """
        # Create the pipeline for inference
        pipeline = self.components.pipeline_cls(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=self.components.transformer,
            scheduler=self.components.scheduler,
        )
        
        return pipeline
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode a video tensor using the VAE.
        
        Args:
            video (torch.Tensor): Video tensor of shape [B, C, F, H, W]
            
        Returns:
            torch.Tensor: Encoded video latents
        """
        # VAE encode the tensor
        with torch.no_grad():
            latents = self.components.vae.encode(video).latent_dist.sample()
            # Apply scaling factor
            scaling_factor = self.components.vae.config.scaling_factor
            latents = latents * scaling_factor
        
        return latents
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text prompt using the text encoder.
        
        Args:
            text (str): Text prompt
            
        Returns:
            torch.Tensor: Encoded text embeddings
        """
        # Tokenize and encode text
        with torch.no_grad():
            inputs = self.components.tokenizer(
                text,
                padding="max_length",
                max_length=self.components.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeddings = self.components.text_encoder(inputs.input_ids.to(self.components.text_encoder.device))[0]
        
        return embeddings
    
    def encode_action(self, action_data: torch.Tensor) -> torch.Tensor:
        """
        Encode action data for the model.
        
        Args:
            action_data (torch.Tensor): Action tensor of shape [T, D]
            
        Returns:
            torch.Tensor: Processed action tensor
        """
        # For now, we just return the tensor as is since we normalize it in the dataset
        # No further processing needed at this stage
        return action_data
    
    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for creating batches from dataset examples.
        
        Args:
            examples (List[Dict[str, Any]]): List of dataset items
            
        Returns:
            Dict[str, torch.Tensor]: Batch of data
        """
        # Extract and stack data from examples
        images = torch.stack([example["image"] for example in examples])
        prompt_embeddings = torch.stack([example["prompt_embedding"] for example in examples])
        encoded_videos = torch.stack([example["encoded_video"] for example in examples])
        action_embeddings = torch.stack([example["action_embedding"] for example in examples])
        
        # Create batch
        batch = {
            "image": images,
            "prompt_embedding": prompt_embeddings,
            "encoded_video": encoded_videos,
            "action_embedding": action_embeddings,
        }
        
        return batch
    
    def compute_loss(self, batch) -> torch.Tensor:
        """
        Compute the training loss for a batch.
        
        Args:
            batch: Batch of data containing image, prompt_embedding, encoded_video, and action_embedding
            
        Returns:
            torch.Tensor: Total loss
        """
        # Extract batch data
        image = batch["image"]
        prompt_embedding = batch["prompt_embedding"]
        encoded_video = batch["encoded_video"]
        action_embedding = batch["action_embedding"]
        
        # Get batch size and device
        batch_size = image.shape[0]
        device = image.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # Add noise to the encoded video
        noise = torch.randn_like(encoded_video)
        noisy_latents = self.components.scheduler.add_noise(encoded_video, noise, timesteps)
        
        # Prepare inputs for the transformer model
        latent_input = torch.cat([noisy_latents, image.unsqueeze(1)], dim=2)
        
        # Forward pass through transformer
        model_output = self.components.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=prompt_embedding,
            actions=action_embedding,
            timestep=timesteps,
            return_dict=True,
        )
        
        # Get video and action predictions
        video_pred = model_output["sample"]
        action_pred = model_output["action"]
        
        # Compute MSE loss for video prediction (noise prediction)
        video_loss = F.mse_loss(video_pred, noise, reduction="mean")
        
        # Compute MSE loss for action prediction
        action_loss = F.mse_loss(action_pred, action_embedding, reduction="mean")
        
        # Total loss is a weighted sum of video and action losses
        # You can adjust these weights based on your requirements
        video_weight = 1.0
        action_weight = self.args.action_loss_weight if hasattr(self.args, "action_loss_weight") else 0.5
        
        total_loss = video_weight * video_loss + action_weight * action_loss
        
        return total_loss
    
    def validation_step(self, sample, pipe) -> List[Tuple[str, Any]]:
        """
        Run validation for a sample.
        
        Args:
            sample: Validation sample (prompt, image, video)
            pipe: Inference pipeline
            
        Returns:
            List[Tuple[str, Any]]: List of validation artifacts (type, value)
        """
        prompt = sample["prompt"]
        image = sample["image"]
        
        # Run inference with the pipeline
        output = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=self.args.guidance_scale,
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            inference_mode="both",  # Generate both video and action
        )
        
        # Extract results
        generated_video = output["video"]
        predicted_actions = output["action_pred"]
        
        # Format the predicted actions for visualization
        action_representation = self._format_actions_for_display(predicted_actions)
        
        # Return validation artifacts
        return [
            ("video", generated_video),
            ("text", f"Actions: {action_representation}")
        ]
    
    def _format_actions_for_display(self, actions: torch.Tensor) -> str:
        """
        Format action tensor for display in validation logs.
        
        Args:
            actions (torch.Tensor): Predicted actions
            
        Returns:
            str: Formatted actions string
        """
        # Extract first few timesteps for display
        actions = actions.detach().cpu().numpy()
        
        # Format for display (customize based on your action representation)
        if len(actions.shape) > 1:
            # Show first few timesteps
            num_display = min(3, actions.shape[0])
            action_str = "\n".join([f"Frame {i}: {actions[i, :4]}..." for i in range(num_display)])
        else:
            action_str = str(actions[:4]) + "..."
            
        return action_str
    
    def save_model(self, output_dir: str, global_step: Optional[int] = None) -> None:
        """
        Save the model to disk.
        
        Args:
            output_dir (str): Directory to save the model
            global_step (int, optional): Current training step
        """
        # Get unwrapped transformer model
        transformer = self.accelerator.unwrap_model(self.components.transformer)
        
        # Create subfolder for this checkpoint if global_step is provided
        if global_step is not None:
            output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Save all components
        if self.args.training_type == "sft":
            # Save full model weights
            transformer.save_pretrained(os.path.join(output_dir, "transformer"))
            
            # Save other components if they exist
            components = {
                "vae": self.components.vae,
                "text_encoder": self.components.text_encoder,
                "tokenizer": self.components.tokenizer,
                "scheduler": self.components.scheduler,
            }
            
            for name, component in components.items():
                if component is not None:
                    component.save_pretrained(os.path.join(output_dir, name))
        
        elif self.args.training_type == "lora":
            # For LoRA training, save adapter weights
            self.components.pipeline_cls.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer.get_adapter_state_dict("default"),
            )
        
        # Save training arguments
        torch.save(self.args.model_dump(), os.path.join(output_dir, "training_args.bin")) 
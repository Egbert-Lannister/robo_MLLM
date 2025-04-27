from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.embeddings import get_3d_rotary_pos_embed

from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel, AutoProcessor
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.transformer.transformer import RoboMultiTransformerModel

import logging
logger = logging.getLogger(__name__)


class RoboMultimodelTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]
    """
    Trainer for RoboMultimodal model.
    """

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        # components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = RoboMultiTransformerModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components
    

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
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
    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Encode the input action into a fixed-length token tensor of shape [168].

        Args:
            action (torch.Tensor): Input tensor with shape [T, D] (e.g., [41, 8]).

        Returns:
            torch.Tensor: Encoded token tensor with shape [168].
        """
        tokenizer = AutoProcessor.from_pretrained("/disk0/home/kuowei/robo_multimodal/action_tokenizer/fast", trust_remote_code=True)

        action_token = tokenizer(action)

        if isinstance(action_token, list):
            action_token = torch.tensor(action_token)

        if not isinstance(action_token, torch.Tensor):
            raise TypeError(f"Unexpected type for action_token: {type(action_token)}")
        
        action_token = action_token.squeeze(0)
        if action_token.dim() != 1:
            raise ValueError(f"Expected 1D token, but got shape {action_token.shape}")

        # Pad or truncate to fixed length 168
        target_length = 168
        current_length = action_token.size(0)

        if current_length < target_length:
            # Padding
            padding = torch.zeros(target_length - current_length, dtype=action_token.dtype, device=action_token.device)
            action_token = torch.cat([action_token, padding], dim=0)
        elif current_length > target_length:
            # Truncating
            action_token = action_token[:target_length]

        return action_token
    
    def batch_action_tokenizer(self, actions_batch: torch.Tensor) -> torch.Tensor:
        # actions_batch: [B, 328]
        outputs = []
        device = actions_batch.device
        for action in actions_batch:
            action = action.detach().cpu().float()
            token = self.encode_action(action)
            if isinstance(token, list):
                token = torch.tensor(token, device=action.device)
            token = token.view(-1)

            outputs.append(token)
        
        return torch.stack(outputs).to(device)  # [B, 168]
    
    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos": [], 
            "prompt_embedding": [], 
            "images": [],
            "encoded_actions": []
        }

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            action_embedding = sample["action_embedding"]

            ret["encoded_videos"].append(encoded_video)
            ret["encoded_actions"].append(action_embedding)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)

        # Stack all tensors
        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["encoded_actions"] = torch.stack(ret["encoded_actions"])

        return ret
    

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
    

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["images"]
        encoded_actions = batch["encoded_actions"]  # This is now a list of tensors with different lengths

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]
        # Shape of encoded_actions: [B, 168]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
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

        # Predict noise and actions
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        transformer_output = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
        predicted_noise = transformer_output[0]
        predicted_actions = transformer_output[1]  # Get predicted actions

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        # Calculate video loss
        video_loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        video_loss = video_loss.mean()

        # Calculate action loss
        predicted_actions_encoded = self.batch_action_tokenizer(predicted_actions)

        predicted_actions_encoded = predicted_actions_encoded.to(dtype=torch.bfloat16)
        encoded_actions = encoded_actions.to(dtype=torch.bfloat16)

        action_loss = F.mse_loss(predicted_actions_encoded, encoded_actions, reduction="mean")

        # Combine losses with equal weights
        total_loss = video_loss + action_loss

        logger.info(f"Video Loss: {video_loss.item()}, Action Loss: {action_loss.item()}")
        logger.info(f"Total Loss: {total_loss.device}, Video Loss: {video_loss.device}, Action Loss: {action_loss.device}") 
        logger.info(f"total_loss type: {total_loss.dtype}, video_loss type: {video_loss.dtype}, action_loss type: {action_loss.dtype}")

        return total_loss
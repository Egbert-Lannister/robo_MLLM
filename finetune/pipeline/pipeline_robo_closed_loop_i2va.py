import torch
from typing import Optional, Union, List, Dict
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DiffusionPipeline
from diffusers.video_processor import VideoProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

from transformer.robo_transformer import RoboTransformer  # Ensure your RoboTransformer model is imported


class RoboClosedLoopTransformerPipeline(DiffusionPipeline):
    """
    A pipeline for training and inference using RoboTransformer for generating both video and action predictions.
    
    Args:
        tokenizer (`T5Tokenizer`): The tokenizer for the input prompts.
        text_encoder (`T5EncoderModel`): The text encoder model to process prompts.
        vae (`AutoencoderKLCogVideoX`): The VAE model for encoding and decoding video latents.
        transformer (`RoboTransformer`): The unified transformer model for generating video and action sequences.
        scheduler (`CogVideoXDDIMScheduler` or `CogVideoXDPMScheduler`): The denoising scheduler for the transformer.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: RoboTransformer,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def encode_prompt(self, prompt: Union[str, List[str]], device: torch.device) -> torch.Tensor:
        # Encode the text prompt using T5 tokenizer and text encoder
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        prompt_embeds = self.text_encoder(inputs["input_ids"].to(device))[0]
        return prompt_embeds

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 49,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preprocess the input image and prepare the latent variables for video generation
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=dtype)
        latents, image_latents = self.vae.prepare_latents(
            image, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents
        )
        return latents, image_latents

    def forward(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        action: Optional[torch.Tensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        device: Optional[torch.device] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[CogVideoXPipelineOutput, Tuple[torch.Tensor]]:
        """
        Generate video and action predictions from image and prompt.

        Args:
            image (`torch.Tensor`): The input image for conditioning the generation.
            prompt (`str` or `List[str]`): The text prompt to guide the generation.
            action (`torch.Tensor`, *optional*): Action data used for training.
            num_frames (`int`): Number of frames to generate for the video.
            num_inference_steps (`int`): Number of denoising steps.
            guidance_scale (`float`): The guidance scale for classifier-free diffusion guidance.
            device (`torch.device`): The device on which the pipeline runs.
            return_dict (`bool`): Whether to return the output as a dictionary.

        Returns:
            CogVideoXPipelineOutput or Tuple: The generated video and action sequence.
        """
        device = device or self._execution_device

        # Encode the text prompt
        prompt_embeds = self.encode_prompt(prompt, device)

        # Prepare latents from the input image
        latents, image_latents = self.prepare_latents(image, batch_size=1, num_frames=num_frames, device=device)

        # Retrieve timesteps for the denoising process
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)

        # Initialize action predictor (for training) or modify to generate actions during inference
        action_predictor = self.transformer.action_predictor if not action else action

        # Denoising loop (for inference)
        for t in timesteps:
            latent_input = torch.cat([latents] * 2, dim=0)  # Classifier-free guidance
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            noise_pred = self.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t,
                return_dict=False,
            )[0]

            # Compute noise for classifier-free guidance
            noise_pred = self._apply_guidance(noise_pred, guidance_scale)

            # Step the scheduler and update latents
            latents = self.scheduler.step(noise_pred, t, latents)[0]

        # Decode the latents back into video frames
        video_frames = self.vae.decode(latents)

        # Generate actions if required
        actions = action_predictor(latents)

        # Return the generated video and actions
        if return_dict:
            return CogVideoXPipelineOutput(frames=video_frames, actions=actions)
        else:
            return (video_frames, actions)

    def _apply_guidance(self, noise_pred: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        # Apply classifier-free guidance to the noise predictions
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)



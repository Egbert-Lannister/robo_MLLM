import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformer.rvaf_transformer import RoboTransformer
# from models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models import AutoencoderKLCogVideoX
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.loaders import CogVideoXLoraLoaderMixin
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from ..utils import logging


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    

class RoboVideoActionFusionPipeline(CogVideoXLoraLoaderMixin, DiffusionPipeline):
    """
    A pipeline for robotic video generation with action fusion for both training and inference.
    
    This pipeline enables:
    1. Training with images, prompts, videos, and action sequences
    2. Inference with multiple modes:
       - Generating videos from images and prompts
       - Predicting action sequences from images and prompts
       - Simultaneously generating videos and predicting actions

    The action information is expected to be a tensor containing robot control data
    such as rotation deltas, gripper state, world vectors, and termination flags.
    
    During inference, use the `inference_mode` parameter to control the output:
    - "video_only": Only generate video frames
    - "action_only": Only predict action sequences
    - "both": Generate both video frames and action sequences (default)
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
        self.vae_scaling_factor_image = self.vae.config.scaling_factor if getattr(self, "vae", None) else 0.7

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        action: Optional[torch.Tensor] = None, # 动作？
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        inference_mode: str = "both", # Add inference_mode parameter with options: "video_only", "action_only", "both"
    ) -> Union[Dict[str, Any], Tuple]:
        
        # 1. 检查输入有效性
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            action=action,  # 检查动作
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )


        
        device = self._execution_device

        # 1. Preprocess image
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)

        # 2. Encode the text prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, guidance_scale > 1, num_videos_per_prompt, device=device
        )
        if guidance_scale > 1:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. Prepare latents and timesteps
        latents, image_latents = self.prepare_latents(image, num_frames, height, width, device)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, guidance_scale, device)

        # 4. Denoising loop
        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents, image_latents], dim=2)

            if guidance_scale > 1:
                latent_input = torch.cat([latent_input] * 2)

            timestep = t.expand(latent_input.shape[0])

            # Use actions tensor if provided
            action_input = action.expand(latent_input.shape[0], *action.shape[1:]) if action is not None else None

            model_output = self.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=prompt_embeds,
                actions=action_input,  # Pass actions to the model
                timestep=timestep,
                return_dict=True,
            )

            noise_pred = model_output['sample']
            action_pred = model_output['action']  # Get action prediction from model output

            # Apply classifier-free guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Also apply guidance to action predictions if needed
                if action_pred is not None:
                    action_pred_uncond, action_pred_cond = action_pred.chunk(2)
                    action_pred = action_pred_uncond + guidance_scale * (action_pred_cond - action_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents)[0]

        # 5. Decode latents to video
        video = self.decode_latents(latents)

        video = self.video_processor.postprocess_video(video, output_type)

        if return_dict:
            if inference_mode == "video_only":
                return {"video": video}
            elif inference_mode == "action_only":
                return {"action_pred": action_pred}
            else:  # Default "both" mode
                return {"video": video, "action_pred": action_pred}

        # Return based on inference_mode
        if inference_mode == "video_only":
            return video
        elif inference_mode == "action_only":
            return action_pred
        else:  # Default "both" mode
            return video, action_pred
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds
    
    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        image = image.unsqueeze(2)  # [B, C, F, H, W]

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

        image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        if not self.vae.config.invert_scale_latents:
            image_latents = self.vae_scaling_factor_image * image_latents
        else:
            # This is awkward but required because the CogVideoX team forgot to multiply the
            # scaling factor during training :)
            image_latents = 1 / self.vae_scaling_factor_image * image_latents

        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Select the first frame along the second dimension
        if self.transformer.config.patch_size_t is not None:
            first_frame = image_latents[:, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents
 
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        frames = self.vae.decode(latents).sample
        return frames
    
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def load_and_preprocess_action(self, action_path, max_frames=49):
        """
        Load and preprocess action data from a JSON file.
        
        Args:
            action_path (str or Path): Path to the JSON file containing action data
            max_frames (int): Maximum number of frames to include
            
        Returns:
            torch.Tensor: Processed action tensor of shape [T, D]
        """
        import json
        from pathlib import Path
        
        action_path = Path(action_path)
        with open(action_path, 'r') as f:
            action_data = json.load(f)
        
        # Convert dict entries to tensor rows
        actions = []
        for item in action_data:
            # Convert all entries to float and flatten into a single list
            row = []
            row.extend(item.get("rotation_delta", [0.0, 0.0, 0.0]))
            row.append(float(item.get("open_gripper", False)))
            row.extend(item.get("world_vector", [0.0, 0.0, 0.0]))
            row.append(float(item.get("terminate_episode", 0.0)))
            actions.append(row)
        
        action_tensor = torch.tensor(actions, dtype=torch.float32)  # Shape: [T, D]
        
        # Truncate or pad to max_frames
        T, D = action_tensor.shape
        if T > max_frames:
            action_tensor = action_tensor[:max_frames]
        elif T < max_frames:
            padding = torch.zeros((max_frames - T, D))
            action_tensor = torch.cat([action_tensor, padding], dim=0)
        
        # Normalize the action tensor to improve training stability
        action_tensor = (action_tensor - action_tensor.mean(dim=0)) / (action_tensor.std(dim=0) + 1e-6)
        
        return action_tensor

    def check_inputs(
        self,
        image,
        prompt,
        height,
        width,
        negative_prompt,
        action,
        callback_on_step_end_tensor_inputs,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        
        if action is not None and not isinstance(action, torch.Tensor):
            raise ValueError(f"`action` has to be of type `torch.Tensor` but is {type(action)}")
        
        # If action is provided, check dimensions
        if action is not None:
            if len(action.shape) != 2:
                raise ValueError(f"Expected action to be a 2D tensor [T, D] but got shape {action.shape}")

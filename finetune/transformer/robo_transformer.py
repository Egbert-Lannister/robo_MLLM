from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
import os
import json


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in model.
    
    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class ActionPredictor(nn.Module):
    """
    A neural network model for predicting robot actions from latent representations.
    
    This model takes latent representations from the transformer model and outputs
    action vectors for robotic control. It uses a multi-layer perceptron architecture
    to map from latent space to action space.
    
    Args:
        latent_dim (int): Dimension of the input latent vectors
        hidden_dim (int): Dimension of the hidden layers
        action_dim (int): Dimension of the output action vectors
        num_layers (int, optional): Number of hidden layers. Defaults to 2.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Store dimensions for future reference
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
    
    def forward(self, latents: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predicts action vectors from latent representations.
        
        Args:
            latents (torch.Tensor): Latent representations from the transformer
                Shape: [batch_size, sequence_length, latent_dim]
            timesteps (torch.Tensor, optional): Timestep embeddings, not used in the base implementation
                but included for compatibility with diffusion models.
        
        Returns:
            torch.Tensor: Predicted action vectors
                Shape: [batch_size, sequence_length, action_dim]
        """
        # Check if we need to reshape the input
        orig_shape = latents.shape
        
        # Flatten if needed (keeping batch dimension)
        if len(orig_shape) > 2:
            latents = latents.reshape(orig_shape[0], -1)
        
        # Run through the MLP
        actions = self.model(latents)
        
        # Reshape output if needed
        if len(orig_shape) > 2:
            actions = actions.reshape(orig_shape[0], -1, self.action_dim)
        
        return actions
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model weights and configuration to disk.
        
        Args:
            save_directory (str): Directory where the model should be saved
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "action_predictor.pt"))
        
        # Save configuration
        config = {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "num_layers": self.num_layers,
        }
        
        with open(os.path.join(save_directory, "action_predictor_config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        """
        Load a pretrained model from disk.
        
        Args:
            pretrained_model_path (str): Path to the directory containing the model files
        
        Returns:
            ActionPredictor: Loaded model instance
        """
        # Load configuration
        with open(os.path.join(pretrained_model_path, "action_predictor_config.json"), "r") as f:
            config = json.load(f)
        
        # Create model with loaded config
        model = cls(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            action_dim=config["action_dim"],
            num_layers=config["num_layers"],
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, "action_predictor.pt")))
        
        return model


class RoboTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        action_predictor: Optional[ActionPredictor] = None,
        **kwargs
    ):
        super().__init__()

        # Existing initialization...
        inner_dim = num_attention_heads * attention_head_dim
        
        # Initialize ActionPredictor if not passed
        if action_predictor is None:
            prompt_dim = text_embed_dim
            image_latent_dim = in_channels  # Assuming image latent dimension matches in_channels
            fused_latent_dim = prompt_dim + image_latent_dim
            action_predictor = ActionPredictor(
                latent_dim=fused_latent_dim,
                hidden_dim=512,
                action_dim=8,
            )
        
        self.action_predictor = action_predictor
        self.register_modules(
            action_predictor=action_predictor,
            # Other model components...
        )
        
        # Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Time embedding layers
        self.time_proj = Timesteps(inner_dim)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)
        
        # Transformer blocks (same as before)
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer norm
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)
        self.proj_out = nn.Linear(inner_dim, out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        task_type: str = 'video',  # New task_type parameter to switch modes
        return_dict: bool = True,
    ):
        """
        Forward method with task_type to select video, action, or joint prediction.

        Parameters:
            task_type (`str`, defaults to `video`):
                - 'video': Video prediction mode
                - 'action': Action prediction mode
                - 'joint': Video and action prediction together
        """
        # Pre-existing code for time embedding, patch embedding, etc.
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep

        # 2. Patch embedding    
        hidden_states = self.patch_embed(hidden_states)

        # 3. Time embedding
        timestep_emb = self.time_embedding(timesteps)

        # 4. Dropout
        hidden_states = self.embedding_dropout(hidden_states)

        # 5. Add time embedding
        hidden_states = hidden_states + timestep_emb

        # 6. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_cond, ofs, image_rotary_emb, attention_kwargs)

        # 7. Final layer norm
        hidden_states = self.norm_final(hidden_states)

        # 8. Project out
        hidden_states = self.proj_out(hidden_states)

        if task_type == 'video':
            output = self._video_forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_cond,
                
            )
        elif task_type == 'action':
            output = self._action_forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
            )
        elif task_type == 'joint':
            video_output = self._video_forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_cond,
                
            )
            action_output = self._action_forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
            )
            output = (video_output, action_output)
        else:
            raise ValueError(f"Invalid task type: {task_type}, must be one of 'video', 'action', or 'joint'")
        
        
        return output
        
    

    def _video_forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        **kwargs
    ):
        
        pass
    

    def _action_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        **kwargs
    ):
        pass
        
   

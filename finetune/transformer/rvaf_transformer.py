import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers import CogVideoXTransformer3DModel


class RoboTransformer(CogVideoXTransformer3DModel):
    def __init__(self, action_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim

        # Define action embedding layer
        self.action_embedding = nn.Linear(action_dim, self.config.time_embed_dim)
        
        # Modify the final output layer to include action prediction
        self.action_predictor = nn.Linear(self.config.inner_dim, action_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,  # Add actions as part of the input
        timestep: Union[int, float, torch.LongTensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if actions is not None:
            # Embed actions and combine with timestep embeddings
            action_emb = self.action_embedding(actions)
            emb = emb + action_emb
        
        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block for video output
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Generate action predictions
        action_pred = self.action_predictor(hidden_states.mean(dim=1))
        
        # 6. Unpatchify for video output
        video_output = self.unpatchify(hidden_states, batch_size, num_frames, height, width)

        # 7. Return results based on return_dict flag
        if return_dict:
            return {
                'sample': video_output,
                'action': action_pred
            }
        else:
            return video_output, action_pred

    def unpatchify(self, hidden_states, batch_size, num_frames, height, width):
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        return output

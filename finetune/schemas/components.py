from typing import Any, Optional

from pydantic import BaseModel
import torch.nn as nn

class Components(BaseModel):
    # pipeline cls
    pipeline_cls: Any = None

    # Tokenizers
    tokenizer: Any = None
    tokenizer_2: Any = None
    tokenizer_3: Any = None

    # Text encoders
    text_encoder: Any = None
    text_encoder_2: Any = None
    text_encoder_3: Any = None

    # Autoencoder
    vae: Any = None

    # Denoiser
    transformer: Any = None
    unet: Any = None

    # Scheduler
    scheduler: Any = None

    # ✅ Action predictor for video-to-action generation
    action_predictor: Optional[nn.Module] = None
    
    class Config:
        arbitrary_types_allowed = True  # ✅ This line fixes the crash
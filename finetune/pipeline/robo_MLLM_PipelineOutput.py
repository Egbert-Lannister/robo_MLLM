from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class RoboMultiPipelineOutput(BaseOutput):
    r"""
    Output class for RoboMulti pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
        actions (`torch.Tensor`):
            List of action outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            action sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, 8)`.
    """

    frames: torch.Tensor = None
    actions: torch.Tensor = None
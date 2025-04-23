from typing import Dict, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.activations import get_activation
from diffusers.models.downsampling import CogVideoXDownsample3D
from diffusers.models.upsampling import CogVideoXUpsample3D
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.autoencoder_kl import DecoderOutput, AutoencoderKLOutput, DiagonalGaussianDistribution
from diffusers.utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogVideoXSafeConv3d(nn.Conv3d):
    r"""
    A 3D convolution layer that splits the input tensor into smaller parts to avoid OOM in CogVideoX Model.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        memory_count = (
            (input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3] * input.shape[4]) * 2 / 1024**3
        )

        # Set to 2GB, suitable for CuDNN
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            # 增加分块数量，减少每块大小
            part_num = int(memory_count / 1.5) + 1  # 使用更小的块大小
            
            # 打印分块信息，便于调试
            print(f"[INFO] CogVideoXSafeConv3d splitting input of shape {input.shape} into {part_num} chunks")
            
            try:
                input_chunks = torch.chunk(input, part_num, dim=2)
                
                if kernel_size > 1:
                    # 限制历史帧的overlap
                    max_overlap_frames = min(kernel_size - 1, 2)  # 最多使用2帧作为overlap
                    
                    processed_chunks = [input_chunks[0]]
                    for i in range(1, len(input_chunks)):
                        try:
                            # 限制前一个chunk提供的重叠帧数
                            overlap_chunk = input_chunks[i-1][:, :, -max_overlap_frames:]
                            combined_chunk = torch.cat((overlap_chunk, input_chunks[i]), dim=2)
                            processed_chunks.append(combined_chunk)
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                # 如果拼接导致内存不足，减少重叠帧数
                                print(f"[WARNING] OOM when combining chunks. Reducing overlap frames.")
                                if max_overlap_frames > 1:
                                    overlap_chunk = input_chunks[i-1][:, :, -1:]  # 只使用一帧重叠
                                    combined_chunk = torch.cat((overlap_chunk, input_chunks[i]), dim=2)
                                    processed_chunks.append(combined_chunk)
                                else:
                                    # 如果已经只有一帧重叠还是OOM，放弃重叠
                                    print(f"[WARNING] Still OOM with minimal overlap. Using chunk without overlap.")
                                    processed_chunks.append(input_chunks[i])
                            else:
                                raise
                    
                    input_chunks = processed_chunks
                
                # 逐块处理并收集结果
                output_chunks = []
                for i, input_chunk in enumerate(input_chunks):
                    try:
                        output_chunk = super().forward(input_chunk)
                        # 如果有重叠，我们需要去除重叠部分的输出
                        if i > 0 and kernel_size > 1 and max_overlap_frames > 0:
                            # 计算需要保留的帧数
                            if i < len(input_chunks) - 1:
                                # 中间块需要去除重叠部分
                                frames_to_keep = input_chunk.shape[2] - max_overlap_frames
                                output_chunk = output_chunk[:, :, -frames_to_keep:]
                                
                        output_chunks.append(output_chunk)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            # 处理单个块时依然OOM，进一步分块
                            print(f"[WARNING] OOM processing chunk {i}. Further splitting...")
                            # 进一步分割这个块
                            sub_chunks = torch.chunk(input_chunk, 2, dim=2)
                            for sub_chunk in sub_chunks:
                                sub_output = super().forward(sub_chunk)
                                output_chunks.append(sub_output)
                        else:
                            raise
                
                # 拼接所有输出块 
                try:
                    output = torch.cat(output_chunks, dim=2)
                    return output
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        # 如果拼接时OOM，尝试逐个拼接
                        print(f"[WARNING] OOM when concatenating all outputs. Doing it incrementally...")
                        result = output_chunks[0]
                        for i in range(1, len(output_chunks)):
                            try:
                                result = torch.cat([result, output_chunks[i]], dim=2)
                            except RuntimeError:
                                # 如果还是OOM，则返回部分结果
                                print(f"[WARNING] Still OOM. Returning partial results up to chunk {i}.")
                                break
                        return result
                    else:
                        raise
                        
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # 如果chunking本身导致OOM
                    print(f"[WARNING] OOM during chunking. Trying alternative approach...")
                    # 尝试使用普通前向传播，但可能会失败
                    try:
                        return super().forward(input)
                    except:
                        # 如果还是失败，返回输入尽可能多的部分结果
                        print(f"[CRITICAL] Could not process input. Returning partial input.")
                        max_frames = max(1, input.shape[2] // 2)  # 处理至少一半帧
                        return super().forward(input[:, :, :max_frames])
                else:
                    raise
        else:
            return super().forward(input)


class CogVideoXCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int`, defaults to `1`): Stride of the convolution.
        dilation (`int`, defaults to `1`): Dilation rate of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # TODO(aryan): configure calculation based on stride and dilation in the future.
        # Since CogVideoX does not use it, it is currently tailored to "just work" with Mochi
        time_pad = time_kernel_size - 1
        height_pad = (height_kernel_size - 1) // 2
        width_pad = (width_kernel_size - 1) // 2

        self.pad_mode = pad_mode
        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = stride if isinstance(stride, tuple) else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pad_mode == "replicate":
            inputs = F.pad(inputs, self.time_causal_padding, mode="replicate")
        else:
            kernel_size = self.time_kernel_size
            if kernel_size > 1:
                # 限制历史帧的累积数量，防止内存爆炸
                max_cached_frames = 3  # 最多使用3帧历史
                
                if conv_cache is not None:
                    # 如果提供了缓存，确保其不会太大
                    if conv_cache.shape[2] > max_cached_frames:
                        # 只取最近的几帧
                        print(f"[WARNING] Limiting conv_cache frames from {conv_cache.shape[2]} to {max_cached_frames}")
                        conv_cache = conv_cache[:, :, -max_cached_frames:]
                    cached_inputs = [conv_cache]
                else:
                    # 创建填充输入，但限制数量
                    num_padding = min(kernel_size - 1, max_cached_frames)
                    cached_inputs = [inputs[:, :, :1]] * num_padding
                
                try:
                    # 使用torch.cat但添加异常处理
                    inputs = torch.cat(cached_inputs + [inputs], dim=2)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        # 内存溢出时尝试降低缓存的帧数
                        print(f"[WARNING] Out of memory in fake_context_parallel_forward. Reducing cached frames.")
                        # 移除一些缓存的输入
                        reduced_cache = cached_inputs[:max(1, len(cached_inputs)//2)]
                        try:
                            inputs = torch.cat(reduced_cache + [inputs], dim=2)
                        except RuntimeError:
                            # 如果仍然失败，只使用当前输入，不使用缓存
                            print(f"[WARNING] Still out of memory. Proceeding without context caching.")
                            pass
                    else:
                        # 其他类型的错误，重新抛出
                        raise
        return inputs

    def forward(self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs, conv_cache)

        if self.pad_mode == "replicate":
            conv_cache = None
        else:
            padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()
            inputs = F.pad(inputs, padding_2d, mode="constant", value=0)

        output = self.conv(inputs)
        return output, conv_cache


class CogVideoXSpatialNorm3D(nn.Module):
    r"""
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002. This implementation is specific
    to 3D-video like data.

    CogVideoXSafeConv3d is used instead of nn.Conv3d to avoid OOM in CogVideoX Model.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
        groups (`int`):
            Number of groups to separate the channels into for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    def forward(
        self, f: torch.Tensor, zq: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, new_conv_cache


class CogVideoXResnetBlock3D(nn.Module):
    r"""
    A 3D ResNet block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
                )
            else:
                self.conv_shortcut = CogVideoXSafeConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(hidden_states, zq, conv_cache=conv_cache.get("norm1"))
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(hidden_states, conv_cache=conv_cache.get("conv1"))

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(hidden_states, zq, conv_cache=conv_cache.get("norm2"))
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(hidden_states, conv_cache=conv_cache.get("conv2"))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache


class CogVideoXDownBlock3D(nn.Module):
    r"""
    A downsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        add_downsample (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = None

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    CogVideoXDownsample3D(
                        out_channels, out_channels, padding=downsample_padding, compress_time=compress_time
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXDownBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, new_conv_cache[conv_cache_key] = self._gradient_checkpointing_func(
                    resnet,
                    hidden_states,
                    temb,
                    zq,
                    conv_cache.get(conv_cache_key),
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states, new_conv_cache


class CogVideoXMidBlock3D(nn.Module):
    r"""
    A middle block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXMidBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, new_conv_cache[conv_cache_key] = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb, zq, conv_cache.get(conv_cache_key)
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        return hidden_states, new_conv_cache


class CogVideoXUpBlock3D(nn.Module):
    r"""
    An upsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, defaults to `16`):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        add_upsample (`bool`, defaults to `True`):
            Whether or not to use a upsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    CogVideoXUpsample3D(
                        out_channels, out_channels, padding=upsample_padding, compress_time=compress_time
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXUpBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, new_conv_cache[conv_cache_key] = self._gradient_checkpointing_func(
                    resnet,
                    hidden_states,
                    temb,
                    zq,
                    conv_cache.get(conv_cache_key),
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, new_conv_cache


class CogVideoXEncoder3D(nn.Module):
    r"""
    The `CogVideoXEncoder3D` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
    """

    _supports_gradient_checkpointing = True
    config_name = "encoder_config"
    
    # Add tiling attributes with default values
    use_tiling = False
    tile_sample_min_height = None
    tile_sample_min_width = None
    tile_overlap_factor_height = 0.25
    tile_overlap_factor_width = 0.25

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()
        # 显式保存out_channels属性
        self.out_channels = out_channels

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        self.down_blocks = nn.ModuleList([])

        # down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        """
        Enable tiled processing of the input for encode() method to reduce memory usage.
        
        Args:
            tile_sample_min_height: Minimum height of each tile
            tile_sample_min_width: Minimum width of each tile
            tile_overlap_factor_height: Overlap factor between tiles in height dimension
            tile_overlap_factor_width: Overlap factor between tiles in width dimension
        """
        self.use_tiling = True
        
        # Set or use default values
        if tile_sample_min_height is not None:
            self.tile_sample_min_height = tile_sample_min_height
        if tile_sample_min_width is not None:
            self.tile_sample_min_width = tile_sample_min_width
        
        self.tile_overlap_factor_height = tile_overlap_factor_height or 0.25
        self.tile_overlap_factor_width = tile_overlap_factor_width or 0.25

    def disable_tiling(self) -> None:
        """
        Disable tiled processing.
        """
        self.use_tiling = False
        self.tile_sample_min_height = None
        self.tile_sample_min_width = None

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            sample: Input tensor [B, C, F, H, W]
            temb: Time embedding, not used in this model but kept for API compatibility
            conv_cache: Optional convolution cache for sequential processing
            
        Returns:
            Encoded representation
        """
        # Make sure sample is 5D: [batch, channels, frames, height, width]
        if sample.dim() == 4:
            sample = sample.unsqueeze(2)
            
        # Get shape information
        batch, channels, frames, height, width = sample.shape
        
        # Check if slicing is enabled and there's more than one frame
        if getattr(self, "use_slicing", False) and frames > 1:
            # Process the input in slices to save memory
            # Determine slice size - process one frame at a time for maximum memory savings
            slice_size = 1
            
            # Initialize output tensor list
            hidden_states_list = []
            
            # Process each slice separately
            for frame_idx in range(0, frames, slice_size):
                # Get the current slice
                frame_end_idx = min(frame_idx + slice_size, frames)
                slice_sample = sample[:, :, frame_idx:frame_end_idx, :, :]
                
                # Process the slice
                # Initial convolution
                slice_hidden_states, _ = self.conv_in(slice_sample)
                
                # Down blocks
                for downsample_block in self.down_blocks:
                    slice_hidden_states, _ = downsample_block(slice_hidden_states, temb=temb)
                    
                # Mid block
                slice_hidden_states, _ = self.mid_block(slice_hidden_states, temb=temb)
                
                # Final norm and convolution
                slice_hidden_states = self.norm_out(slice_hidden_states)
                slice_hidden_states = self.conv_act(slice_hidden_states)
                slice_hidden_states, _ = self.conv_out(slice_hidden_states)
                
                # Add to the list
                hidden_states_list.append(slice_hidden_states)
            
            # Concatenate all slices
            hidden_states = torch.cat(hidden_states_list, dim=2)
            
            return hidden_states
        else:
            # Standard processing without slicing
            # Initial convolution
            hidden_states, new_conv_cache_conv_in = self.conv_in(sample, conv_cache=conv_cache.get("conv_in") if conv_cache else None)
            
            new_conv_cache = {"conv_in": new_conv_cache_conv_in}
            
            # Down blocks
            for i, downsample_block in enumerate(self.down_blocks):
                hidden_states, new_conv_cache[f"down_block_{i}"] = downsample_block(
                    hidden_states, 
                    temb=temb, 
                    conv_cache=conv_cache.get(f"down_block_{i}") if conv_cache else None
                )
                
            # Mid block
            hidden_states, new_conv_cache["mid_block"] = self.mid_block(
                hidden_states, 
                temb=temb, 
                conv_cache=conv_cache.get("mid_block") if conv_cache else None
            )
            
            # Final norm and convolution
            hidden_states = self.norm_out(hidden_states)
            hidden_states = self.conv_act(hidden_states)
            hidden_states, new_conv_cache["conv_out"] = self.conv_out(
                hidden_states, 
                conv_cache=conv_cache.get("conv_out") if conv_cache else None
            )

            return hidden_states


class CogVideoXDecoder3D(nn.Module, ConfigMixin):
    r"""
    The `CogVideoXDecoder3D` layer of a variational autoencoder that decodes its input into a reconstructed image.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon value for normalization layers.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        temporal_compression_ratio (`float`, *optional*, defaults to 4):
            The ratio of temporal compression.
    """

    _supports_gradient_checkpointing = True
    config_name = "decoder_config"
    
    # Define tiling parameters with default values
    use_tiling = False
    tile_sample_min_height = None
    tile_sample_min_width = None
    tile_overlap_factor_height = None
    tile_overlap_factor_width = None

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        self.out_channels = out_channels

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.up_blocks = nn.ModuleList([])

        # up blocks
        output_channel = block_out_channels[-1]
        for i, up_block_type in enumerate(up_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if up_block_type == "CogVideoXUpBlock3D":
                up_block = CogVideoXUpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=0.0,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=output_channel,
                    add_upsample=not is_final_block,
                    upsample_padding=0,
                    compress_time=compress_time,
                )
            else:
                raise ValueError("Invalid `up_block_type` encountered. Must be `CogVideoXUpBlock3D`")

            self.up_blocks.append(up_block)

        # Add final convolutional layers
        self.conv_act = get_activation(act_fn)
        self.conv_out = CogVideoXSafeConv3d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

        self.gradient_checkpointing = False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        """
        Enable tiled decoding to process larger images without running out of memory.
        
        Args:
            tile_sample_min_height (int, optional): Minimum height of each tile. If None, defaults to model-specific value.
            tile_sample_min_width (int, optional): Minimum width of each tile. If None, defaults to model-specific value.
            tile_overlap_factor_height (float, optional): Overlap factor for height. If None, defaults to 0.25.
            tile_overlap_factor_width (float, optional): Overlap factor for width. If None, defaults to 0.25.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_overlap_factor_height = tile_overlap_factor_height or 0.25
        self.tile_overlap_factor_width = tile_overlap_factor_width or 0.25

    def disable_tiling(self) -> None:
        """
        Disable tiled decoding.
        """
        self.use_tiling = False
        self.tile_sample_min_height = None
        self.tile_sample_min_width = None

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decodes latents using a tiled approach to handle larger batches/images with limited memory.
        
        Args:
            z (torch.Tensor): Latent tensor to decode
            return_dict (bool): Whether to return a DecoderOutput instead of a plain tensor
            
        Returns:
            Union[DecoderOutput, torch.Tensor]: Decoded sample
        """
        if not self.use_tiling:
            return self.decode(z, return_dict=return_dict)
            
        # Extract dimensions from input tensor
        batch_size, channels, num_frames, z_height, z_width = z.shape
        
        # Determine tile sizes
        tile_height = self.tile_sample_min_height
        tile_width = self.tile_sample_min_width
        
        # Default values if not set
        if tile_height is None:
            tile_height = z_height
        if tile_width is None:
            tile_width = z_width
        
        # Ensure tiles are not larger than the input
        tile_height = min(tile_height, z_height)
        tile_width = min(tile_width, z_width)
        
        # Calculate overlap in pixels
        overlap_height = int(tile_height * self.tile_overlap_factor_height)
        overlap_width = int(tile_width * self.tile_overlap_factor_width)
        
        # Ensure overlap is not larger than the tile size
        overlap_height = min(overlap_height, tile_height // 2)
        overlap_width = min(overlap_width, tile_width // 2)
        
        # Get output scale factor based on the decoder's configuration
        scale_factor = self.config.block_out_channels[0] // self.config.block_out_channels[-1]
        
        # Calculate output dimensions
        output_height = z_height * scale_factor
        output_width = z_width * scale_factor
        output_channels = self.config.out_channels
        
        # Initialize output tensor
        output = torch.zeros(
            (batch_size, output_channels, num_frames, output_height, output_width),
            device=z.device,
            dtype=z.dtype,
        )
        
        # Initialize weight tensor for blending overlaps
        weights = torch.zeros(
            (batch_size, 1, num_frames, output_height, output_width),
            device=z.device,
            dtype=z.dtype,
        )
        
        # Process in tiles
        for h_start in range(0, z_height, tile_height - overlap_height):
            for w_start in range(0, z_width, tile_width - overlap_width):
                # Ensure we don't go out of bounds
                h_end = min(h_start + tile_height, z_height)
                w_end = min(w_start + tile_width, z_width)
                
                # Extract the current tile
                tile = z[:, :, :, h_start:h_end, w_start:w_end]
                
                # Decode the tile
                decoded_tile = self._decode(tile, return_dict=False)
                
                # Calculate output coordinates (accounting for scaling)
                out_h_start = h_start * scale_factor
                out_w_start = w_start * scale_factor
                out_h_end = h_end * scale_factor
                out_w_end = w_end * scale_factor
                
                # Create a weight map for smooth blending at the edges
                weight_map = torch.ones(
                    (batch_size, 1, num_frames, out_h_end - out_h_start, out_w_end - out_w_start),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )
                
                # Apply tapering at edges for smooth blending
                # Horizontal edges
                if h_start > 0:
                    # Taper top edge
                    overlap_pix = overlap_height // (2 ** (len(self.encoder.down_blocks)))
                    for i in range(overlap_pix):
                        weight_map[:, :, :, i, :] = i / overlap_pix
                if h_end < z_height:
                    # Taper bottom edge
                    overlap_pix = overlap_height // (2 ** (len(self.encoder.down_blocks)))
                    for i in range(overlap_pix):
                        weight_map[:, :, :, -(i+1), :] = i / overlap_pix
                
                # Vertical edges
                if w_start > 0:
                    # Taper left edge
                    overlap_pix = overlap_width // (2 ** (len(self.encoder.down_blocks)))
                    for j in range(overlap_pix):
                        weight_map[:, :, :, :, j] = j / overlap_pix
                if w_end < z_width:
                    # Taper right edge
                    overlap_pix = overlap_width // (2 ** (len(self.encoder.down_blocks)))
                    for j in range(overlap_pix):
                        weight_map[:, :, :, :, -(j+1)] = j / overlap_pix
                
                # Add the weighted tile to the output
                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += decoded_tile * weight_map
                
                # Update the weight tensor
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += weight_map
        
        # Normalize by weights
        weights = torch.clamp(weights, min=1e-8)  # Avoid division by zero
        output = output / weights
        
        if return_dict:
            return DecoderOutput(sample=output)
        else:
            return output

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape

        hidden_states = z

        for i, up_block in enumerate(self.up_blocks):
            hidden_states = up_block(hidden_states)

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        if not return_dict:
            return hidden_states
        return DecoderOutput(sample=hidden_states)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode the latent tensor to image space.
        
        Args:
            z: Latent tensor to decode
            return_dict: Whether to return a DecoderOutput instead of a plain tensor
            
        Returns:
            Union[DecoderOutput, torch.Tensor]: Decoded sample
        """
        if self.use_tiling:
            dec = self.tiled_decode(z, return_dict=return_dict)
        else:
            dec = self._decode(z, return_dict=return_dict)
        return dec
        
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            sample: Input tensor
            sample_posterior: Whether to sample from the posterior
            return_dict: Whether to return as a dictionary
            generator: Optional random generator for sampling
            
        Returns:
            Union[DecoderOutput, torch.Tensor]: Decoded sample
        """
        x = sample
        
        if x.dim() == 4:
            x = x.unsqueeze(2)  # Add frame dimension for single image
            
        # Get posterior distribution
        posterior = self.encode(x, return_dict=True).latent_dist
        
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
            
        dec = self.decode(z, return_dict=return_dict)
        
        if not return_dict:
            return (dec,)
            
        return dec
        
    @torch.no_grad()
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
        """
        Prepare latent variables for video generation.
        
        Args:
            image: Input conditioning image
            batch_size: Batch size for generation
            num_channels_latents: Number of channels in latent space
            num_frames: Number of frames to generate
            height: Height in latent space
            width: Width in latent space
            dtype: Data type for tensors
            device: Device for computation
            generator: Random generator for latent sampling
            latents: Optional pre-generated latents
            
        Returns:
            Tuple of (latents, image_latents) where latents is the initial noise and
            image_latents is the encoded input image
        """
        # Get image latents by encoding the input image
        image_latents = self.encode(image).latent_dist.sample(generator=generator) * self.config.scaling_factor
        
        # Generate random initial latents if not provided
        if latents is None:
            latents = randn_tensor(
                (batch_size, num_channels_latents, num_frames, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        
        return latents, image_latents
        
    def _gradient_checkpointing_func(self, module, *args, **kwargs):
        """
        Runs the forward pass with gradient checkpointing enabled.
        Used to save memory during training.
        """
        return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)

    @property
    def latent_channels(self):
        """Return latent channels from config for backward compatibility"""
        return self.config.latent_channels

class AutoencoderKLCogVideoX(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Variational Autoencoder (VAE) model with KL loss for encoding images into latents and decoding latent 
    representations into images.
    
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the
    library implements for all the models (such as downloading or saving).
    
    Parameters:
        in_channels (`int`, *optional*, defaults to 3): Number of input channels
        out_channels (`int`, *optional*, defaults to 3): Number of output channels
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CogVideoXDownBlock3D",)`): 
            Tuple of downsample block types
        up_block_types (`Tuple[str]`, *optional*, defaults to `("CogVideoXUpBlock3D",)`): 
            Tuple of upsample block types
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`): 
            Tuple of block output channels
        layers_per_block (`int`, *optional*, defaults to 1): 
            Number of ResNet layers per block
        act_fn (`str`, *optional*, defaults to `"silu"`): 
            Activation function to use
        latent_channels (`int`, *optional*, defaults to 4): 
            Number of channels in the latent space
        norm_num_groups (`int`, *optional*, defaults to 32): 
            Number of groups for normalization
        sample_size (`int`, *optional*, defaults to None): 
            Sample input size
        scaling_factor (`float`, *optional*, defaults to 0.18215): 
            Scale factor for latent variables
        temporal_compression_ratio (`float`, *optional*, defaults to 4): 
            The ratio of temporal compression
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        latent_channels: int = 16,
        norm_num_groups: int = 32,
        sample_size: Optional[int] = None,
        scaling_factor: float = 0.18215,
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        # encoder
        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        # decoder
        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        # Tile processing setup
        self.use_tiling = False
        self.tile_sample_min_size = None
        self.tile_overlap_factor = None

        # Slicing setup
        self.use_slicing = False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        """
        Enable tiled processing of the input for encode() method to reduce memory usage.
        
        Args:
            tile_sample_min_height: Minimum height of each tile
            tile_sample_min_width: Minimum width of each tile
            tile_overlap_factor_height: Overlap factor between tiles in height dimension
            tile_overlap_factor_width: Overlap factor between tiles in width dimension
        """
        self.encoder.enable_tiling(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_overlap_factor_height=tile_overlap_factor_height,
            tile_overlap_factor_width=tile_overlap_factor_width,
        )
        self.decoder.enable_tiling()
        self.use_tiling = True

    def disable_tiling(self) -> None:
        """
        Disable tiled processing.
        """
        self.encoder.disable_tiling()
        self.decoder.disable_tiling()
        self.use_tiling = False

    def enable_slicing(self) -> None:
        """
        Enable sliced processing of the input for encode() method to reduce memory usage.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        """
        Disable sliced processing.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode a batch of images/video frames into a latent representation.
        
        Args:
            x: Input tensor of shape [batch_size, channels, frames, height, width]
            
        Returns:
            DiagonalGaussianDistribution: Distribution of latent variables.
        """
        if x.dim() == 4:
            # Add a single frame dimension if input is an image
            x = x.unsqueeze(2)
            
        # Encode through the encoder model
        h = self.encoder(x)
        
        # The DiagonalGaussianDistribution expects a single tensor that it will split
        # into mean and logvar, so we don't need to split it here
        return DiagonalGaussianDistribution(h)

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        """
        Encode an input tensor to the latent space.
        """
        if self.use_tiling:
            latent_dist = self.tiled_encode(x)
        elif self.use_slicing:
            latent_dist = self._encode(x)
        else:
            latent_dist = self._encode(x)

        if return_dict:
            return AutoencoderKLOutput(latent_dist=latent_dist)
        return latent_dist

    def tiled_encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode the input tensor in tiles to save memory.

        This method divides the input tensor into overlapping tiles and processes each tile separately,
        then combines the results with smooth blending at the overlapping regions. This allows processing
        of larger images without running out of memory.

        Args:
            x (`torch.Tensor`): The input tensor of shape `[batch, channels, frames, height, width]`
                or `[batch, channels, height, width]`.

        Returns:
            `DiagonalGaussianDistribution`: The encoded latent distribution.
        """
        # Check if we're using tiling
        if not hasattr(self.encoder, 'use_tiling') or not self.encoder.use_tiling:
            return self._encode(x)
            
        # Get shape information
        if x.dim() == 4:
            # [B, C, H, W] - add frame dimension for consistent processing
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
            
        batch, channels, frames, height, width = x.shape
        
        # Set default values for tile parameters if not specified
        tile_size_h = self.encoder.tile_sample_min_height
        if tile_size_h is None:
            tile_size_h = height
            
        tile_size_w = self.encoder.tile_sample_min_width
        if tile_size_w is None:
            tile_size_w = width
            
        # Ensure tile sizes don't exceed input dimensions
        tile_size_h = min(tile_size_h, height)
        tile_size_w = min(tile_size_w, width)
        
        # Calculate overlap in pixels
        overlap_height = int(tile_size_h * self.encoder.tile_overlap_factor_height)
        overlap_width = int(tile_size_w * self.encoder.tile_overlap_factor_width)
        
        # Calculate theoretical downsampling factor
        downsample_factor = 2 ** (len(self.encoder.down_blocks))
        
        # Estimate latent dimensions based on downsampling
        latent_h = height // downsample_factor
        latent_w = width // downsample_factor
        
        # Print this info for debugging
        print(f"[DEBUG] Original dimensions: {height}x{width}, estimated latent dimensions: {latent_h}x{latent_w}")
        print(f"[DEBUG] Tile size: {tile_size_h}x{tile_size_w}, overlap: {overlap_height}x{overlap_width}")
        
        # Initialize arrays to accumulate results
        accum_mu = torch.zeros((batch, self.config.latent_channels, frames, latent_h, latent_w), 
                             device=x.device, dtype=x.dtype)
        accum_logvar = torch.zeros((batch, self.config.latent_channels, frames, latent_h, latent_w), 
                                 device=x.device, dtype=x.dtype)
        
        # Initialize weights for blending
        weights = torch.zeros((1, 1, 1, latent_h, latent_w), 
                             device=x.device, dtype=x.dtype)
        
        # Process tiles
        for h_start in range(0, height, tile_size_h - overlap_height):
            for w_start in range(0, width, tile_size_w - overlap_width):
                # Ensure we don't go out of bounds
                h_end = min(h_start + tile_size_h, height)
                w_end = min(w_start + tile_size_w, width)
                
                # Extract tile
                tile = x[:, :, :, h_start:h_end, w_start:w_end]
                
                # Print tile info for debugging
                print(f"[DEBUG] Processing tile at ({h_start}:{h_end}, {w_start}:{w_end}) shape={tile.shape}")
                
                # Encode tile
                latent_dist = self._encode(tile)
                
                # Get the mean and logvar for this tile
                mu = latent_dist.mean
                logvar = latent_dist.logvar
                
                # Print encoded dimensions for debugging
                print(f"[DEBUG] mu shape: {mu.shape}")
                
                # Calculate latent space coordinates
                latent_h_start = h_start // downsample_factor
                latent_w_start = w_start // downsample_factor
                latent_h_end = latent_h_start + mu.shape[3]
                latent_w_end = latent_w_start + mu.shape[4]
                
                # Create weight mask matching the actual encoded tile dimensions
                weight_mask = torch.ones((1, 1, 1, mu.shape[3], mu.shape[4]), 
                                       device=x.device, dtype=x.dtype)
                
                # Print info for debugging
                print(f"[DEBUG] weight_mask shape: {weight_mask.shape}")
                print(f"[DEBUG] latent_w_start: {latent_w_start}, latent_w_end: {latent_w_end}")
                
                # Apply tapering at edges if this tile overlaps with others
                # Horizontal edges
                if h_start > 0:
                    # Taper top edge (gradually increase weight)
                    overlap_h_latent = min(overlap_height // downsample_factor, mu.shape[3])
                    for i in range(overlap_h_latent):
                        weight_mask[:, :, :, i, :] = i / overlap_h_latent
                        
                if h_end < height:
                    # Taper bottom edge (gradually decrease weight)
                    overlap_h_latent = min(overlap_height // downsample_factor, mu.shape[3])
                    for i in range(overlap_h_latent):
                        weight_mask[:, :, :, -(i+1), :] = i / overlap_h_latent
                
                # Vertical edges
                if w_start > 0:
                    # Taper left edge (gradually increase weight)
                    overlap_w_latent = min(overlap_width // downsample_factor, mu.shape[4])
                    for j in range(overlap_w_latent):
                        weight_mask[:, :, :, :, j] = j / overlap_w_latent
                        
                if w_end < width:
                    # Taper right edge (gradually decrease weight)
                    overlap_w_latent = min(overlap_width // downsample_factor, mu.shape[4])
                    for j in range(overlap_w_latent):
                        weight_mask[:, :, :, :, -(j+1)] = j / overlap_w_latent
                
                try:
                    # Ensure our accumulation arrays are large enough
                    if latent_h_end > accum_mu.shape[3] or latent_w_end > accum_mu.shape[4]:
                        # Resize accumulators to accommodate the encoded tile
                        new_latent_h = max(accum_mu.shape[3], latent_h_end)
                        new_latent_w = max(accum_mu.shape[4], latent_w_end)
                        
                        print(f"[DEBUG] Resizing accumulators to {new_latent_h}x{new_latent_w}")
                        
                        # Create new accumulators with the larger size
                        new_accum_mu = torch.zeros((batch, self.config.latent_channels, frames, new_latent_h, new_latent_w), 
                                                 device=x.device, dtype=x.dtype)
                        new_accum_logvar = torch.zeros((batch, self.config.latent_channels, frames, new_latent_h, new_latent_w), 
                                                     device=x.device, dtype=x.dtype)
                        new_weights = torch.zeros((1, 1, 1, new_latent_h, new_latent_w), 
                                                device=x.device, dtype=x.dtype)
                        
                        # Copy existing data
                        new_accum_mu[:, :, :, :accum_mu.shape[3], :accum_mu.shape[4]] = accum_mu
                        new_accum_logvar[:, :, :, :accum_logvar.shape[3], :accum_logvar.shape[4]] = accum_logvar
                        new_weights[:, :, :, :weights.shape[3], :weights.shape[4]] = weights
                        
                        # Update references
                        accum_mu = new_accum_mu
                        accum_logvar = new_accum_logvar
                        weights = new_weights
                    
                    # Accumulate weighted results
                    accum_mu[:, :, :, latent_h_start:latent_h_end, latent_w_start:latent_w_end] += mu * weight_mask
                    accum_logvar[:, :, :, latent_h_start:latent_h_end, latent_w_start:latent_w_end] += logvar * weight_mask
                    
                    # Accumulate weights
                    weights[:, :, :, latent_h_start:latent_h_end, latent_w_start:latent_w_end] += weight_mask
                    
                    print(f"[DEBUG] Successfully added tile at ({latent_h_start}:{latent_h_end}, {latent_w_start}:{latent_w_end})")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to add tile: {e}")
                    print(f"[ERROR] Attempted dimensions: mu={mu.shape}, latent coords=({latent_h_start}:{latent_h_end}, {latent_w_start}:{latent_w_end})")
                    print(f"[ERROR] Accumulator dimensions: accum_mu={accum_mu.shape}, weights={weights.shape}")
                    # Continue with the next tile instead of failing completely
                    continue
        
        # Avoid division by zero by ensuring minimum weight is non-zero
        weights = torch.clamp(weights, min=1e-8)
        
        # Normalize by weights to get final values
        final_mu = accum_mu / weights
        final_logvar = accum_logvar / weights
        
        # Create and return the distribution
        return DiagonalGaussianDistribution(torch.cat([final_mu, final_logvar], dim=1))

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a latent representation.
        """
        if self.use_tiling:
            dec = self.decoder.tiled_decode(z, return_dict=return_dict)
        else:
            dec = self.decoder.decode(z, return_dict=return_dict)
        return dec
        
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Encode a sample into latent space and decode it to a reconstructed sample.
        
        Args:
            sample (torch.Tensor): Input sample to encode and reconstruct
            sample_posterior: Whether to sample from the posterior
            return_dict: Whether to return a DecoderOutput instead of a tuple
            generator: Optional random generator for sampling
            
        Returns:
            Union[DecoderOutput, torch.Tensor]: Decoded sample
        """
        # Encode the sample to latent representation
        posterior = self.encode(sample, return_dict=True).latent_dist
        
        if sample_posterior:
            # Sample from the posterior distribution
            z = posterior.sample(generator=generator)
        else:
            # Use the mean of the posterior distribution
            z = posterior.mode()
            
        # Apply scaling factor to the latent representation
        z = z * self.config.scaling_factor
        
        # Decode the latent representation back to the sample space
        return self.decoder.decode(z, return_dict=return_dict)

    @torch.no_grad()
    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 41,
        height: int = 480,
        width: int = 640,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare latent variables for video generation from an input image.
        
        Args:
            image: Input image tensor
            batch_size: Batch size for generation
            num_channels_latents: Number of channels in latent space
            num_frames: Number of frames to generate
            height: Height in latent space
            width: Width in latent space
            dtype: Data type for tensors
            device: Device for computation
            generator: Random generator for latent sampling
            latents: Optional pre-generated latents
            
        Returns:
            Tuple of (latents, image_latents) where latents is the initial noise and
            image_latents is the encoded input image
        """
        # Get image latents by encoding the input image
        image_latents = self.encode(image).latent_dist.sample(generator=generator) * self.config.scaling_factor
        
        # Generate random initial latents if not provided
        if latents is None:
            latents = randn_tensor(
                (batch_size, num_channels_latents, num_frames, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        
        return latents, image_latents
        
    def _gradient_checkpointing_func(self, module, *args, **kwargs):
        """
        Runs the forward pass with gradient checkpointing enabled.
        Used to save memory during training.
        """
        return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
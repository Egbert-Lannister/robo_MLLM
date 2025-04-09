import logging
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


# 新增动作数据加载函数
def load_actions(action_path: Path) -> List[Path]:
    with open(action_path, "r", encoding="utf-8") as file:
        return [action_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]

##########  preprocessors  ##########


def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image

'''
def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count < max_num_frames, we repeat the last frame until max_num_frames.
      2. If video frame count >= max_num_frames, we do evenly spaced sampling.
      3. Decord will handle resizing to (height, width) if explicitly provided.

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)

    print(f"[DEBUG] --> preprocess_video_with_resize: video_path={video_path},"
          f" max_num_frames={max_num_frames}, target_size=({height}, {width})")

    # 打开视频并检查帧数
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    print(f"[DEBUG] Video loaded. Total frames in file: {video_num_frames}")

    # 如果视频文件帧数 < max_num_frames，重复最后一帧
    if video_num_frames < max_num_frames:
        frames = video_reader.get_batch(list(range(video_num_frames)))
        print(f"[DEBUG] Raw frames shape: {frames.shape} (expect [T, H, W, C])")

        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)

        print(f"[DEBUG] After repeating last frame {num_repeats} times: {frames.shape} (should have {max_num_frames} frames now)")
        frames = frames.float().permute(0, 3, 1, 2).contiguous()
        print(f"[DEBUG] Final shape after permute to [F, C, H, W]: {frames.shape}")

        return frames

    else:
        # 帧数 >= max_num_frames 时，做等间隔采样
        indices = []
        # 避免出现整除问题时取到比 max_num_frames 多的帧，这里用一个更安全的写法
        step = video_num_frames / max_num_frames
        for i in range(max_num_frames):
            index = int(round(i * step))
            # 避免因为 round 导致越界，这里 clamp 一下
            index = min(index, video_num_frames - 1)
            indices.append(index)

        frames = video_reader.get_batch(indices)
        print(f"[DEBUG] Chosen frame indices (length={len(indices)}): {indices}")
        print(f"[DEBUG] Sampled frames shape before float/permute: {frames.shape}")

        frames = frames[:max_num_frames].float()  # 保险操作，截断到 max_num_frames
        frames = frames.permute(0, 3, 1, 2).contiguous()
        print(f"[DEBUG] Final shape after permute to [F, C, H, W]: {frames.shape}")

        return frames
'''

def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames

def preprocess_video_with_buckets(
    video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file.
        resolution_buckets: List of tuples (num_frames, height, width) representing
            available resolution buckets.

    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    resolution_buckets = [bucket for bucket in resolution_buckets if bucket[0] <= video_num_frames]
    if len(resolution_buckets) == 0:
        raise ValueError(f"video frame count in {video_path} is less than all frame buckets {resolution_buckets}")

    nearest_frame_bucket = min(
        resolution_buckets,
        key=lambda bucket: video_num_frames - bucket[0],
        default=1,
    )[0]
    frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))
    frames = video_reader.get_batch(frame_indices)
    frames = frames[:nearest_frame_bucket].float()
    frames = frames.permute(0, 3, 1, 2).contiguous()

    nearest_res = min(resolution_buckets, key=lambda x: abs(x[1] - frames.shape[2]) + abs(x[2] - frames.shape[3]))
    nearest_res = (nearest_res[1], nearest_res[2])
    frames = torch.stack([resize(f, nearest_res) for f in frames], dim=0)

    return frames



def preprocess_actions(action_path: Path, max_num_frames: int) -> torch.Tensor:
    """
    Loads and processes action data to match the target length.

    Args:
        action_path: Path to the action file.
        max_num_frames: Target number of time steps for the action sequence.

    Returns:
        torch.Tensor: Processed action tensor with shape [target_length, feature_dim]
    """
    with open(action_path, 'r') as f:
        action_data = json.load(f)
    
    # Convert action data to tensor
    action_tensor = torch.tensor([list(d.values()) for d in action_data], dtype=torch.float32)
    
    # Get current length of action sequence
    current_length = action_tensor.shape[0]
    
    if current_length < max_num_frames:
        # Repeat the last action to reach target_length
        last_action = action_tensor[-1:]
        num_repeats = max_num_frames - current_length
        repeated_actions = last_action.repeat(num_repeats, 1)
        action_tensor = torch.cat([action_tensor, repeated_actions], dim=0)
    elif current_length > max_num_frames:
        # Downsample the action sequence
        indices = list(range(0, current_length, current_length // max_num_frames))
        action_tensor = action_tensor[indices[:max_num_frames]]
    
    return action_tensor
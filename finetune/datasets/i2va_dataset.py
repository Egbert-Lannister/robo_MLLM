import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_prompts,
    load_videos,
    load_images,
    load_actions,
    load_images_from_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
    preprocess_video_with_buckets,
    preprocess_actions,
)

if TYPE_CHECKING:
    from finetune.trainer import Trainer

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseI2VADataset(Dataset):
    """
    Base dataset class for Image-to-Video-to-Action (I2VA) training.

    This dataset loads prompts, videos, corresponding conditioning images, and action data for I2VA training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        action_column (str): Path to file containing action data paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        action_column: str,
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.trainer = trainer  # 设置 trainer
        print(f"Trainer type: {type(trainer)}")  # 确保是正确的 Trainer 子类
        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        if image_column is not None:
            self.images = load_images(data_root / image_column)
        else:
            self.images = load_images_from_videos(self.videos)
        self.actions = load_actions(data_root / action_column)  # 新增动作数据加载

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.encode_action = trainer.encode_action
        # print(f"encode_video: {self.encode_video}")
        # print(f"encode_text: {self.encode_text}")
        # print(f"encode_action: {self.encode_action}")


        # Check if number of prompts matches number of videos and images and actions
        if not (len(self.videos) == len(self.prompts) == len(self.images) == len(self.actions)):
            raise ValueError(
                f"Expected length of prompts, videos, images, and actions to be the same but found {len(self.prompts)=}, {len(self.videos)=}, {len(self.images)=}, and {len(self.actions)=}. Please ensure that the number of caption prompts, videos, images, and actions match in your dataset."
            )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if all image files exist
        if any(not path.is_file() for path in self.images):
            raise ValueError(
                f"Some image files were not found. Please ensure that all image files exist in the dataset directory. Missing file: {next(path for path in self.images if not path.is_file())}"
            )

        # Check if all action files exist
        if any(not path.is_file() for path in self.actions):
            raise ValueError(
                f"Some action files were not found. Please ensure that all action files exist in the dataset directory. Missing file: {next(path for path in self.actions if not path.is_file())}"
            )
        
    def load_and_preprocess_action(self, action_path: Path, max_timesteps: int) -> torch.Tensor:
        """
        Load action data from a JSON file and preprocess it to a tensor of shape (T, D),
        where T = max_num_frames and D is the flattened dimension of all action keys.
        """
        print(f"Calling encode_action in {self.__class__.__name__}")
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

        action_tensor = torch.tensor(actions, dtype=torch.float32)  # Shape: (T, D)

        # Truncate or pad to max_timesteps
        T, D = action_tensor.shape
        if T > max_timesteps:
            action_tensor = action_tensor[:max_timesteps]
        elif T < max_timesteps:
            padding = torch.zeros((max_timesteps - T, D))
            action_tensor = torch.cat([action_tensor, padding], dim=0)

        # Apply action transform
        return self.action_transform(action_tensor)

    def __len__(self) -> int:
        return len(self.videos)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        video = self.videos[index]
        image = self.images[index]
        action = self.actions[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        action_embeddings_dir = cache_dir / "action_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        action_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")
        action_embedding_path = action_embeddings_dir / (action.stem + f"_{self.max_num_frames}.safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            _, image = self.preprocess(None, self.images[index])
            image = self.image_transform(image)
        else:
            frames, image = self.preprocess(video, image)
            frames = frames.to(self.device)
            image = image.to(self.device)
            image = self.image_transform(image)
            frames = self.video_transform(frames)

            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = image.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)
        
        
        # print(f"Loading action data for index={index}")
        # print(f"Calling encode_action for action data")
        if action_embedding_path.exists():
            action_embedding = load_file(action_embedding_path)["action_embedding"]
            logger.debug(f"Loaded action embedding from {action_embedding_path}", main_process_only=False)
        else:
            action_data = self.load_and_preprocess_action(action, self.max_num_frames)
            action_embedding = self.encode_action(action_data)
            action_embedding = action_embedding.to("cpu")
            save_file({"action_embedding": action_embedding}, action_embedding_path)
            logger.info(f"Saved action embedding to {action_embedding_path}", main_process_only=False)

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "action_embedding": action_embedding,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }
    

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")
    
    def action_transform(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")


class I2VADatasetWithResize(BaseI2VADataset):
    """
    A dataset class for image-to-video-to-action generation that resizes inputs to fixed dimensions.

    This class preprocesses videos, images, and actions by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width
    - Actions are loaded and preprocessed

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print(f"[DEBUG] Trainer type: {type(self.trainer)}")  # Should print the correct Trainer subclass

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
    
    @override
    def action_transform(self, action: torch.Tensor) -> torch.Tensor:
        # Normalize or otherwise transform action data if needed
        # Here, we simply standardize to [-1, 1] range assuming input values are float
        # You can customize this normalization logic based on dataset statistics
        # print("call the function", action)
        # print()
        # print((action - action.mean(dim=0)) / (action.std(dim=0) + 1e-6))
        # return (action - action.mean(dim=0)) / (action.std(dim=0) + 1e-6)
        return action


class I2VADatasetWithBuckets(BaseI2VADataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)

    @override
    def load_and_preprocess_action(self, action_path: Path) -> torch.Tensor:
        return preprocess_actions(action_path, self.trainer.args.max_num_frames)


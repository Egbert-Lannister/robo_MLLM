import os
import logging
import torch
from diffusers import CogVideoXImageToVideoActionPipeline
from diffusers import ActionPredictor
from diffusers.utils import export_to_video, load_image
import json
import numpy as np

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "/disk0/home/kuowei/debug_chectpoint"
TEST_DATA_ROOT = "/disk0/home/kuowei/bridge_finetune_gt_1000_test_20"  # 测试数据集目录
OUTPUT_DIR = "/disk0/home/kuowei/action_debug"

NUM_FRAMES = 41      # 视频帧数
WIDTH = 640          # 视频宽度
HEIGHT = 480         # 视频高度
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
FPS = 16
SEED = 42


def load_test_data(test_data_root):
    prompts_path = os.path.join(test_data_root, "prompts.txt")
    images_txt_path = os.path.join(test_data_root, "images.txt")
    
    # 读取 prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # 读取图片路径（images.txt 里每一行形如 "images/00500.jpg"）
    with open(images_txt_path, "r", encoding="utf-8") as f:
        image_files = [line.strip() for line in f if line.strip()]
    
    # 直接将 test_data_root 与 image_files 拼接，而不再手动加 "images_folder"
    # 这样 "images/00500.jpg" 会变成 "/home/kuowei/bridge_finetune_gt_test_20/images/00500.jpg"
    image_paths = [os.path.join(test_data_root, img) for img in image_files]
    
    return prompts, image_paths

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 创建 actions 和 videos 子文件夹
    actions_dir = os.path.join(OUTPUT_DIR, "actions")
    videos_dir = os.path.join(OUTPUT_DIR, "videos")
    os.makedirs(actions_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    logging.info("Loading fine-tuned model from %s", MODEL_PATH)
    
    # 1. 加载 ActionPredictor
    action_predictor = ActionPredictor.from_pretrained(
        "/disk0/home/kuowei/debug_chectpoint/action_predictor"
    ).to(torch.bfloat16)
    
    # 2. 初始化管道
    pipe = CogVideoXImageToVideoActionPipeline.from_pretrained(
        MODEL_PATH, 
        action_predictor=action_predictor,
        torch_dtype=torch.bfloat16
    )
    
    # 3. 设备与内存优化
    pipe.to("cuda")
    pipe.vae.enable_slicing()  # ✅ 正确调用 VAE 方法
    pipe.vae.enable_tiling()
    
    # 4. 加载测试数据
    prompts, image_paths = load_test_data(TEST_DATA_ROOT)
    test_samples = list(zip(prompts, image_paths))[:20]
    
    # 5. 执行推理
    for i, (prompt, image_path) in enumerate(test_samples):
        logging.info("Generating video for sample %d: %s", i, prompt)
        
        image = load_image(image_path)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipe(
                prompt=prompt,
                image=image,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator("cuda").manual_seed(SEED)
            )

        # 获取视频帧并转换格式
        video_frames = output.frames[0]  # 假设取第一个批次

        # 6. 保存结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存 video 到 videos 子文件夹
        output_video_path = os.path.join(videos_dir, f"sample_{base_name}.mp4")
        export_to_video(video_frames, output_video_path, fps=FPS)
        logging.info(f"Video saved to: {output_video_path}")
        
        # 保存 action 到 actions 子文件夹
        output_action_path = os.path.join(actions_dir, f"sample_{base_name}_actions.json")
        with open(output_action_path, "w") as action_file:
            json.dump(output.actions, action_file, indent=4)
        logging.info(f"Actions saved to: {output_action_path}")


if __name__ == "__main__":
    main()

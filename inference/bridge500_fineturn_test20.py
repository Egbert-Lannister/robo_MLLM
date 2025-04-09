import os
import logging
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "/home/kuowei/cogvideo_diffuser_fineturn_transformer/cogvideo_bridge500_merged_transformer_checkpoint"
TEST_DATA_ROOT = "/home/kuowei/bridge_finetune_gt_test_20"  # 测试数据集目录
OUTPUT_DIR = "/home/kuowei/bridge_finetune_gt_test_20/videos_2"

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
    logging.info("Loading fine-tuned model from %s", MODEL_PATH)
    
    # 加载精调模型
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    # 如需开启 CPU offload，则注释掉 pipe.to("cuda") 或注释下行 enable_sequential_cpu_offload
    pipe.enable_sequential_cpu_offload()
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # pipe.to("cuda")
    
    # 加载测试数据
    prompts, image_paths = load_test_data(TEST_DATA_ROOT)
    test_samples = list(zip(prompts, image_paths))[:20]
    
    for i, (prompt, image_path) in enumerate(test_samples):
        logging.info("Generating video for sample %d: %s", i, prompt)
        
        # 在此检查实际路径
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        image = load_image(image_path)
        
        # 进行推理
        output = pipe(
            prompt=prompt,
            image=image,
            height=HEIGHT,
            width=WIDTH,
            num_videos_per_prompt=1,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED)
        )
        video_frames = output.frames[0]
        
        # 文件名以输入图片的基础名加上前缀 sample_
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_video_path = os.path.join(OUTPUT_DIR, f"sample_{base_name}.mp4")
        
        export_to_video(video_frames, output_video_path, fps=FPS)
        logging.info("Saved video to %s", output_video_path)


if __name__ == "__main__":
    main()

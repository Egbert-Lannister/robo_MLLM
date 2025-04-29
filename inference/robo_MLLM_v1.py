import os
import sys
import json
import logging
import torch
from diffusers.utils import export_to_video, load_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.pipeline.pipeline_robo_MLLM import RoboMultiImageToVideoPipeline
from finetune.transformer.transformer import RoboMultiTransformerModel
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from transformers import T5EncoderModel, AutoTokenizer

# ========= 配置 =========
MODEL_PATH = "/disk0/home/kuowei/Robo_MLLM_v1"
TEST_DATA_ROOT = "/disk0/home/kuowei/bridge_finetune_gt_1000_test_20"
OUTPUT_DIR = "/disk0/home/kuowei/action_debug"

NUM_FRAMES = 41
WIDTH = 640
HEIGHT = 480
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
FPS = 16
SEED = 42
DTYPE = torch.bfloat16  # 推理默认使用 bfloat16
DEVICE = torch.device("cuda")

logging.basicConfig(level=logging.INFO)

# ========= 加载测试数据 =========
def load_test_data(test_data_root):
    prompts_path = os.path.join(test_data_root, "prompts.txt")
    images_txt_path = os.path.join(test_data_root, "images.txt")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    with open(images_txt_path, "r", encoding="utf-8") as f:
        image_files = [line.strip() for line in f if line.strip()]

    image_paths = [os.path.join(test_data_root, img) for img in image_files]
    return prompts, image_paths

# ========= 保存动作 =========
def save_actions(actions: torch.Tensor, save_path: str):
    actions = actions.cpu().tolist()
    actions_list = []
    for action in actions:
        actions_list.append({
            "rotation_delta": action[:3],
            "open_gripper": action[3] > 0.0,
            "world_vector": action[4:7],
            "terminate_episode": float(action[7]),
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(actions_list, f, indent=4)

# ========= 手动处理 action_transformer =========
def materialize_module(module: torch.nn.Module, device: torch.device):
    for name, child in module.named_children():
        materialize_module(child, device)

    # 处理模型参数
    for param in module.parameters(recurse=False):
        if param.device.type == "meta":
            with torch.no_grad():
                new_param = torch.empty_like(param, device=device)
                new_param.copy_(param.to(dtype=new_param.dtype, device=device))
                param.data = new_param
            print(f"Materialized {param} to {device}")

    # 处理缓冲区
    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer.device.type == "meta":
            with torch.no_grad():
                new_buffer = torch.empty_like(buffer, device=device)
                new_buffer.copy_(buffer.to(dtype=new_buffer.dtype, device=device))
                module._buffers[buffer_name] = new_buffer
            print(f"Materialized {buffer_name} to {device}")


# ========= 主程序 =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    actions_dir = os.path.join(OUTPUT_DIR, "actions")
    videos_dir = os.path.join(OUTPUT_DIR, "videos")
    os.makedirs(actions_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    logging.info("Loading fine-tuned RoboMultiTransformer model from %s", MODEL_PATH)

    # 加载模型
    transformer = RoboMultiTransformerModel.from_pretrained(
        MODEL_PATH, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    scheduler = CogVideoXDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    # Materialize action_transformer
    if hasattr(transformer, "action_transformer"):
        logging.info("Materializing action_transformer to device...")
        materialize_module(transformer.action_transformer, DEVICE)

    # 初始化 pipeline
    pipeline = RoboMultiImageToVideoPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    pipeline.to(DEVICE, dtype=DTYPE)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    pipeline.enable_sequential_cpu_offload()

    pipeline.transformer.action_transformer.to_empty(device=DEVICE)

    prompts, image_paths = load_test_data(TEST_DATA_ROOT)
    test_samples = list(zip(prompts, image_paths))[:20]

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    for i, (prompt, image_path) in enumerate(test_samples):
        logging.info("enerating video and actions for sample %d: %s", i, prompt)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = load_image(image_path)

        with torch.autocast(device_type="cuda", dtype=DTYPE):
            output = pipeline(
                image=image,
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                num_videos_per_prompt=1,
                generator=generator,
                output_type="pil",
                return_dict=True,
            )

        video_frames = output.frames[0]
        actions = output.actions[0]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_video_path = os.path.join(videos_dir, f"sample_{base_name}.mp4")
        output_action_path = os.path.join(actions_dir, f"sample_{base_name}.json")

        export_to_video(video_frames, output_video_path, fps=FPS)
        save_actions(actions, output_action_path)

        logging.info("Saved video to %s", output_video_path)
        logging.info("Saved actions to %s", output_action_path)

if __name__ == "__main__":
    main()

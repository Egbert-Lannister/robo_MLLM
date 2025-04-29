import os
import sys
import json
import torch
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 保持不变
from finetune.transformer.transformer import RoboMultiTransformerModel

# ====== 配置 ======
model_dir = "/disk0/home/kuowei/Robo_MLLM_v1/transformer"
transformer_model_path = model_dir
safetensors_file = os.path.join(model_dir, "diffusion_pytorch_model-00001-of-00005.safetensors")  # 其中一个权重文件
index_file = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
# ==================

# Step 1. 检查 checkpoint index 里是否有 action_transformer 的 key
print("\n📦 Step 1: 检查 checkpoint index 里是否有 action_transformer keys")
with open(index_file, "r") as f:
    index = json.load(f)

keys = index["weight_map"].keys()
action_keys = [key for key in keys if "action_transformer" in key]

if action_keys:
    print(f"✅ 找到 {len(action_keys)} 个 action_transformer 权重名：")
    for key in action_keys:
        print(f"   - {key}")
else:
    print("❌ 没有找到任何 action_transformer 权重！")
    exit(1)

# Step 2. 加载一个 safetensors 文件，检查 action_transformer 的权重是不是正常的 Tensor
print("\n📦 Step 2: 检查 safetensors 文件中是否存在 action_transformer 的数据")

# 查找所有 safetensors 文件
all_files = os.listdir(model_dir)
safetensors_files = [f for f in all_files if f.endswith(".safetensors")]

if not safetensors_files:
    raise FileNotFoundError(f"❌ 没有找到任何 .safetensors 文件 in {model_dir}！")
else:
    print(f"🔎 找到以下 safetensors 文件:")
    for f in safetensors_files:
        print(f"   - {f}")

# 检查所有 safetensors 文件中是否存在 action_transformer 的权重
# 检查所有 safetensors 文件中是否存在 action_transformer 的权重
found_action_transformer = False
for safetensors_file in safetensors_files:
    safetensors_path = os.path.join(model_dir, safetensors_file)
    
    try:
        # 加载 safetensors 文件
        tensor_file = load_file(safetensors_path)
        
        # 查找包含 "action_transformer" 的键
        tensor_keys = tensor_file.keys()
        found_keys = [k for k in tensor_keys if "action_transformer" in k]
        
        if found_keys:
            found_action_transformer = True
            print(f"\n✅ 在 {safetensors_file} 里找到了 {len(found_keys)} 个 action_transformer tensor：")
            for key in found_keys:
                print(f"   - {key}")
                
                # 获取对应的 tensor 值并打印出来
                tensor_value = tensor_file[key]
                
                # 检查 tensor 是否是有效的，并且输出其数值
                if isinstance(tensor_value, torch.Tensor):
                    print(f"   ✅ tensor 的形状为: {tensor_value.shape}")
                    print(f"   ✅ tensor 的数值：\n{tensor_value}")
                else:
                    print(f"   ❌ {key} 不是有效的 tensor，无法输出数值")
        else:
            print(f"❌ {safetensors_file} 中没有任何 action_transformer 的 tensor！")
    
    except Exception as e:
        print(f"⚠️ 加载 {safetensors_file} 时发生错误: {e}")

if found_action_transformer:
    print("\n✅ action_transformer 权重已成功找到！")
else:
    print("\n❌ 没有找到任何 action_transformer 权重！")

# Step 3. 正确加载 transformer model
print("\n🚀 Step 3: 加载 RoboMultiTransformerModel from_pretrained...")
model = RoboMultiTransformerModel.from_pretrained(
    transformer_model_path,
    subfolder="",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    device_map=None,
)

# Step 4. 检查 action_transformer 里面有没有 meta tensor
print("\n🔍 Step 4: 检查 action_transformer 模块的参数 device 和数值")

meta_params = []
normal_params = []

# 查看模型中的所有 action_transformer 参数
for n, p in model.action_transformer.named_parameters():
    if p.device.type == "meta":
        meta_params.append(n)
    else:
        # 检查是否为有效的数值（即非meta，并且其值是否为 NaN）
        if torch.isnan(p).any():
            print(f"❌ {n} 含有 NaN 数值！")
        else:
            normal_params.append(n)

if normal_params:
    print(f"✅ 正常加载了 {len(normal_params)} 个参数，比如：")
    for name in normal_params[:5]:  # 只列一部分
        print(f"   - {name}")
if meta_params:
    print(f"\n❌ 发现 {len(meta_params)} 个 meta tensor，列表如下：")
    for name in meta_params:
        print(f"   - {name}")
    print("\n⚠️ 推理时一定会崩溃，需要 materialize 这些参数到 device！")

    # 这里做 materialize 操作
    print("\n🔧 进行 materialize 操作...")
    for name in meta_params:
        param = dict(model.action_transformer.named_parameters())[name]
        if param.device.type == "meta":
            with torch.no_grad():
                # 使用 to_empty() 转移到正确的设备
                param.data = param.to_empty(device=param.device)
                print(f"✅ 已将 {name} 转移到设备 {param.device}")
else:
    print("\n✅ action_transformer 所有参数都在正常 device 上，推理可以继续！")

print("\n🎯 检查完成！")

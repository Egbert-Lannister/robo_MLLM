from transformers import AutoProcessor

# 下载预训练的 tokenizer 源代码
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


import os
import numpy as np
from datasets import load_dataset
import tiktoken
from sklearn.model_selection import train_test_split

# 输出目录
out_dir = os.path.dirname(__file__)
train_bin_file = os.path.join(out_dir, "train_3.bin")
val_bin_file = os.path.join(out_dir, "val_3.bin")

# 加载 OpenWebText 数据
print("Loading OpenWebText dataset...")
dataset = load_dataset("openwebtext")

train_data = dataset["train"]
n_total = len(train_data)
print(f"Total samples in OpenWebText: {n_total}")

# 采样 5%
sample_ratio = 0.03
n_sample = int(n_total * sample_ratio)
np.random.seed(42)  # 保证复现
indices = np.random.choice(n_total, n_sample, replace=False)
sampled_data = train_data.select(indices)

print(f"Sampled {n_sample} documents ({sample_ratio*100:.1f}%)")

# 划分 train / val（90% / 10%）
train_texts, val_texts = train_test_split(
    sampled_data["text"], test_size=0.1, random_state=42
)

# 初始化 GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")

def encode(texts):
    """将文本列表编码为 GPT-2 token id"""
    ids = []
    for t in texts:
        ids.extend(enc.encode_ordinary(t))
        ids.append(enc.eot_token)  # 句尾加 <|endoftext|>
    return np.array(ids, dtype=np.uint16)

print("Encoding train split...")
train_ids = encode(train_texts)
print("Encoding val split...")
val_ids = encode(val_texts)

# 保存到 bin 文件
print(f"Saving train ({len(train_ids)} tokens) to {train_bin_file}")
train_ids.tofile(train_bin_file)

print(f"Saving val ({len(val_ids)} tokens) to {val_bin_file}")
val_ids.tofile(val_bin_file)

print("✅ Done! 3% OpenWebText dataset ready.")

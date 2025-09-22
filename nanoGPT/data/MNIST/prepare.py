from torchvision import datasets, transforms
import numpy as np
import pickle
import os

# 下载 MNIST
mnist = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
 
# 创建保存目录
os.makedirs('data/MNIST', exist_ok=True)

# 提取数据
all_data = []
for img, label in mnist:
    # img: [1,28,28], 转为 (784,) 数组
    arr = (img.numpy().flatten() * 255).astype(int).tolist()  # 像素 0-255
    # 这里可以选择把 label 加到序列末尾，比如 arr + [label]
    all_data.extend(arr)

# 构建词表（0-255）
vocab_size = 256
stoi = {i: i for i in range(vocab_size)}
itos = {i: i for i in range(vocab_size)}

# 保存词表
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('data/MNIST/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# 划分训练/验证
n = len(all_data)
train_data = np.array(all_data[:int(n*0.9)], dtype=np.uint16)
val_data = np.array(all_data[int(n*0.9):], dtype=np.uint16)

train_data.tofile('data/MNIST/train.bin')
val_data.tofile('data/MNIST/val.bin')

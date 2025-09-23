import torch

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# data
dataset = 'openwebtext-3%'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 20000
lr_decay_iters = 20000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimizer selection
optimizer_type = 'muon'  # 'adamw' or 'muon'

# adamw optimizer settings
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# muon optimizer settings (only used if optimizer_type = 'muon')
muon_momentum = 0.95
muon_lr = 0.02
use_muon_for_hidden_only = True  # recommended to use Muon only for hidden layers

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
min_lr = 6e-5

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# initialization
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

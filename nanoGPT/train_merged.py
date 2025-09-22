#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (DDP).

English quickstart:
-------------------
Single GPU (debug):
    $ python train_merged.py --batch_size=32 --compile=False

DDP on 1 node with 4 GPUs:
    $ torchrun --standalone --nproc_per_node=4 train_merged.py

DDP across 2 nodes with 4 GPUs each (example master IP 123.456.123.456):
    # master
    $ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=123.456.123.456 --master_port=1234 train_merged.py
    # worker
    $ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr=123.456.123.456 --master_port=1234 train_merged.py
If your cluster does not have Infiniband, prepend: NCCL_IB_DISABLE=1

中文快速开始：
-------------
#### 单卡运行
    python train_merged.py config/train_merged.py
#### 分布式运行（单机8卡示例）
    torchrun --standalone --nproc_per_node=8 train_merged.py config/train_merged.py

Optimizer selection (compatibility):
------------------------------------
- Upper-file style (benchmark/runner):
    * Use `--optimizer_name=ADAMW` (or ADAM / SGD / MUON / MUON_WITH_AUX_ADAM).
    * Or set `optimizer_class` directly in a config file.
- Lower-file style (experimenter/Muon-first):
    * Use `--optimizer_type=adamw` or `--optimizer_type=muon`.
    * For Muon, you can set `--use_muon_for_hidden_only=True` to apply Muon to hidden
      weights and AdamW to embeddings/lm_head via MuonWithAuxAdam.

Logging:
--------
- Set env var LOG_CSV="results/benchmark_runs.csv" to append a CSV log.
- Set RUN_NAME for per-run labeling. Optionally enable Weights & Biases via wandb_log.

Notes:
------
- This file merges and preserves the functionality of both provided scripts:
  * Generic optimizer factory + CSV logging + epochs->iters bridge (upper file).
  * Muon-first options, hidden-only Muon, checkpointing, eval cadence, and rich
    diagnostics (grad norms, condition numbers, sharpness) (lower file).
"""

import csv
import inspect
import json
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Try to make "opts.muon" importable (upper-file layout)
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
except Exception:
    pass

from optimizer_config import initialize_optimizer, _pick_optimizer_class_and_groups
# Import GPT model
from model import GPTConfig, GPT

print("running train_merged.py")

# -----------------------------------------------------------------------------
# Default config values (can be overridden via configurator.py or CLI args there)
# -----------------------------------------------------------------------------
out_dir = 'out_openwebtext'
eval_interval = 10
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' | 'resume' | 'gpt2*'

# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data
dataset = 'openwebtext-3%'
trainingset = 'train.bin'     # upper-file default
validationset = 'val.bin'     # upper-file default
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# seeding
seed = 2  # upper-file default; we add per-rank offset for DDP

# Optimizer selection knobs (both styles supported)
optimizer_type = 'muon'     # 'adamw' or 'muon' (lower-file style). If None, use optimizer_name/class.
optimizer_name = None     # 'ADAM'|'ADAMW'|'SGD'|'MUON'|'MUON_WITH_AUX_ADAM' (upper-file style)
optimizer_class = None    # Direct class, optional
param_groups = None       # Upper-file style param group configs; see initialize_optimizer()

# training horizon & hyperparams
epochs = None             # If set, overrides max_iters
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
momentum = 0.95
use_muon_for_hidden_only = True  # if True: hidden weights via Muon, others via Adam
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 20000
min_lr = 6e-5

# DDP settings
backend = 'nccl'  # 'nccl' | 'gloo' etc.

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# CSV logger (upper file)
log_csv = os.environ.get("LOG_CSV")  # e.g., "results/benchmark_runs.csv"
folder = os.path.dirname(log_csv)
run_name = os.environ.get("RUN_NAME")  # optional per-run label

# -----------------------------------------------------------------------------
# Configurator hook (Karpathy-style); can override any of the above from a .py
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# Post-config adjustments
if epochs is not None:
    max_iters = int(epochs)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _resolve_memmap_filename(split: str, data_dir: str) -> str:
    """Return the first existing memmap filename for the given split.

    Priority order:
    1) trainingset/validationset (config)
    2) *_3.bin alternative (lower file) if present
    3) standard 'train.bin' / 'val.bin'
    """
    preferred = trainingset if split == 'train' else validationset
    candidates = [preferred]

    base, ext = os.path.splitext(preferred)
    if not base.endswith('_3'):
        candidates.append(base + '_3' + ext)

    # explicit lower-file names
    if split == 'train':
        candidates += ['train_3.bin', 'train.bin']
    else:
        candidates += ['val_3.bin', 'val.bin']

    for fname in candidates:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return path
    # fallback
    return os.path.join(data_dir, preferred)

# === Extra CSV helpers for diagnostics ===
_metric_csv_writers = {}  # name -> (fh, writer)

def _setup_metric_csv(name: str, fieldnames):
    """Create/open a CSV at results/<name>.csv with given fieldnames."""
    global _metric_csv_writers
    if name in _metric_csv_writers:
        return _metric_csv_writers[name]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.csv")
    file_exists = os.path.exists(path)
    fh = open(path, "a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    _metric_csv_writers[name] = (fh, writer)
    return fh, writer

def _csv_log_metric(name: str, row: dict):
    """Append one row to results/<name>.csv; creates file on first use."""
    fh, writer = _setup_metric_csv(name, fieldnames=list(row.keys()))
    writer.writerow(row)


def _csv_logger_setup():
    if not (master_process and log_csv):
        return None, None
    os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(log_csv)
    fh = open(log_csv, "a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=[
        "timestamp", "run_name", "optimizer_name", "seed", "dataset",
        "iter", "split", "loss", "learning_rate", "mfu_percent", "dt_ms"
    ])
    if not file_exists:
        writer.writeheader()
    return fh, writer


def _csv_log(writer, split, it, loss_val=None, lr_val=None, mfu_val=None, dt_ms=None):
    if writer is None:
        return
    writer.writerow({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name or "",
        "optimizer_name": (optimizer_name or (optimizer_class.__name__ if optimizer_class else "")),
        "seed": seed,
        "dataset": dataset,
        "iter": int(it),
        "split": split,
        "loss": float(loss_val) if loss_val is not None else "",
        "learning_rate": float(lr_val) if lr_val is not None else "",
        "mfu_percent": float(mfu_val) if isinstance(mfu_val, (int, float)) else "",
        "dt_ms": float(dt_ms) if dt_ms is not None else "",
    })


# -----------------------------------------------------------------------------
# DDP init and environment
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# seed (preserve upper-file 'seed' but add per-rank offset for DDP like lower-file)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loader (memmap)
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

def get_batch(split: str):
    # pick best-available memmap file for this split
    path = _resolve_memmap_filename(split, data_dir)
    mm = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(mm) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((mm[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((mm[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Model init / resume / GPT-2 init
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# try to read vocab size from meta
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif isinstance(init_from, str) and init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop model block size if desired
if block_size < getattr(model, 'config', GPTConfig(vocab_size=50304, block_size=1024)).block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# GradScaler (be permissive to PyTorch versions)
try:
    scaler = torch.amp.GradScaler(device_type=device_type, enabled=(dtype == 'float16'))
except TypeError:
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# pick optimizer class and param groups (deferred until after definitions)
optimizer_class_obj, param_groups_config, chosen_opt_name = _pick_optimizer_class_and_groups(weight_decay, learning_rate, beta1, beta2, optimizer_class, param_groups, optimizer_name, ddp, momentum, optimizer_type, use_muon_for_hidden_only)
optimizer_name = optimizer_name or chosen_opt_name  # keep for logging

# create optimizer
optimizer = initialize_optimizer(model, optimizer_class_obj, param_groups_config, device_type, ddp=ddp)

# resume optimizer state if resuming
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free

# -----------------------------------------------------------------------------
# Compile and DDP wrap
# -----------------------------------------------------------------------------
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0+

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Eval helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return float(min_lr + coeff * (learning_rate - min_lr))

# -----------------------------------------------------------------------------
# Optional W&B
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

csv_fh, csv_writer = _csv_logger_setup()

criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

while True:
    # LR schedule
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg['learning_rate'] = lr

    # periodic evaluation (lower-file cadence) + checkpointing
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": float(losses['train']),
                "val/loss": float(losses['val']),
                "learning_rate": lr,
                "mfu": running_mfu * 100.0,
            })
        _csv_log(csv_writer, "train_eval", iter_num, loss_val=float(losses['train']), lr_val=lr, mfu_val=(running_mfu*100 if running_mfu>=0 else None))
        _csv_log(csv_writer, "val_eval", iter_num, loss_val=float(losses['val']), lr_val=lr, mfu_val=(running_mfu*100 if running_mfu>=0 else None))

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = float(losses['val'])
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if master_process:
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # fwd/bwd with grad accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')  # prefetch next batch
        scaler.scale(loss).backward()

    # clip gradients
    # print("Clipping Gradients")
    if grad_clip and grad_clip > 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # print("Gradients Clipped")

    # step optimizer
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    # timing/logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        _csv_log(csv_writer, "train_iter", iter_num, loss_val=lossf, lr_val=lr, mfu_val=(running_mfu*100 if running_mfu>=0 else None), dt_ms=dt*1000.0)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if csv_fh:
    csv_fh.close()

if ddp:
    destroy_process_group()

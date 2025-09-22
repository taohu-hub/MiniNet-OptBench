#!/bin/bash
export NCCL_BLOCKING_WAIT=1  # 避免 NCCL 网络等待问题
export NCCL_IB_DISABLE=1
# 节点信息
MASTER_ADDR="10.200.250.35"   # master 节点 IP
MASTER_PORT=29500            # master 节点端口（随便一个没被占用的）
NNODES=2                     # 节点总数
NODE_RANK=0                # 当前节点的 rank (0 或 1)
GPUS_PER_NODE=4              # 每台机器的 GPU 数量

# 启动 torchrun
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py config/train_shakespeare_char.py


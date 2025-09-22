#!/bin/bash
ARCH=CNN #Supported: CNN or nanoGPT
DEVICE=cuda:7
DATASET=cifar10
EPOCHS=5
SEEDS=2
BATCH=12
WEIGHT_DECAY=0
BETAS=0.9,0.999
EPS=1e-8
COMPILE=True

TRAIN_SCRIPT="python -m run_benchmark"

mkdir -p results
export LOG_CSV=results${DATASET}/bench_runs.csv   # all runs append to one CSV

# Optimizer-specific LR grids
ADAM_LRS="0.0003 0.001 0.01"
ADAMW_LRS="0.0003 0.001 0.01"
SGD_LRS="0.0003 0.001 0.01 0.001"
MUON_LRS="0.0003 0.001 0.01 0.02"
MWAA_LRS="0.0003 0.001 0.01 0.02"
COMMON_MOMS="0.9" #"0.0 0.9 0.95"   # (Adam/AdamW ignore momentum; harmless to pass)

# SGD
for LR in $SGD_LRS; do
  for M in $COMMON_MOMS; do
    RUN_NAME=SGD_lr${LR}_m${M}
    RUN_NAME=$RUN_NAME $TRAIN_SCRIPT \
      --ARCH=$ARCH --device=$DEVICE --dataset=$DATASET --optimizers=SGD \
      --epochs=$EPOCHS --seeds=$SEEDS --batch_size=$BATCH \
      --learning_rate=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
      --momentum=$M
  done
done

# # MUON_WITH_AUX_ADAM
# for LR in $MWAA_LRS; do
#   RUN_NAME=MWAA_lr${LR}_m0.95
#   RUN_NAME=$RUN_NAME $TRAIN_SCRIPT \
#     --ARCH=$ARCH --device=$DEVICE --dataset=$DATASET --optimizers=MUON_WITH_AUX_ADAM \
#     --epochs=$EPOCHS --seeds=$SEEDS --batch_size=$BATCH \
#     --learning_rate=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
#     --momentum=0.95
# done


#ADAM
for OPT in ADAM ADAMW; do
  for LR in $ADAM_LRS; do
    RUN_NAME=${OPT}_lr${LR}
    RUN_NAME=$RUN_NAME $TRAIN_SCRIPT \
      --ARCH=$ARCH --device=$DEVICE --dataset=$DATASET --optimizers=$OPT \
      --epochs=$EPOCHS --seeds=$SEEDS --batch_size=$BATCH \
      --learning_rate=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
      --momentum=0.0
  done
done

# MUON
for LR in $MUON_LRS; do
  for M in 0.95 0.99; do
    RUN_NAME=MUON_lr${LR}_m${M}
    RUN_NAME=$RUN_NAME $TRAIN_SCRIPT \
      --ARCH=$ARCH --device=$DEVICE --dataset=$DATASET --optimizers=MUON \
      --epochs=$EPOCHS --seeds=$SEEDS --batch_size=$BATCH \
      --learning_rate=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
      --momentum=$M
  done
done




#!/bin/bash
ARCH=PINN # Supported: CNN, PINN or nanoGPT
DEVICE=auto
DATASET=cifar10
EPOCHS=12000
SEEDS=2
BATCH=12
WEIGHT_DECAY=0
BETAS=0.9,0.999
EPS=1e-8
COMPILE=True

TRAIN_SCRIPT="python -m run_benchmark"

mkdir -p "results/${DATASET}"
export LOG_CSV="results/${DATASET}/bench_runs.csv"   # all runs append to one CSV

if [ "$ARCH" = "PINN" ]; then
  # Datasets for PINN: use PDE names
  if [ "$DATASET" = "cifar10" ]; then
    DATASETS=(convection reaction wave)
  else
    DATASETS=($DATASET)
  fi

  # Per-optimizer LR grids for PINN
  ADAM_LRS_PINN="0.0003 0.001"
  SGD_LRS_PINN="0.0003 0.001 0.01"
  MUON_LRS_PINN="0.005 0.01 0.02"
  MWAA_LRS_PINN="0.005 0.01 0.02"  # Muon With Aux Adam
  SGD_MOMS_PINN="0.0 0.9"
  MUON_MOM="0.95"

  # Split betas into two tokens: e.g., "0.9 0.999"
  BETAS_ARR=(${BETAS//,/ })

  for PDE in "${DATASETS[@]}"; do
    RESULTS_DIR="results/pinn/${PDE}"
    mkdir -p "${RESULTS_DIR}"
    export LOG_CSV="${RESULTS_DIR}/bench_runs.csv"

    # Map 'auto' to GPU 0 for PINN script; run_experiment maps to CPU if CUDA not available
    if [ "$DEVICE" = "auto" ]; then
      PINN_DEVICE="0"
    else
      PINN_DEVICE="$DEVICE"
    fi

    # Provide default PDE parameters when absent
    case "$PDE" in
      convection)
        PDE_PARAMS=(--pde_params beta 10)
        ;;
      reaction)
        PDE_PARAMS=(--pde_params rho 5)
        ;;
      wave)
        PDE_PARAMS=(--pde_params beta 10)
        ;;
      *)
        PDE_PARAMS=()
        ;;
    esac

    # ADAM
    for LR in $ADAM_LRS_PINN; do
      for S in $(seq 1 $SEEDS); do
        CMD=(python PINN/run_experiment.py
             --seed=$S
             --pde=$PDE
             --opt=adam
             --opt_params lr $LR betas ${BETAS_ARR[@]} eps $EPS weight_decay $WEIGHT_DECAY
             --epochs=$EPOCHS
             --device=$PINN_DEVICE)
        CMD+=("${PDE_PARAMS[@]}")
        "${CMD[@]}"
      done
    done

    # SGD
    for LR in $SGD_LRS_PINN; do
      for M in $SGD_MOMS_PINN; do
        for S in $(seq 1 $SEEDS); do
          CMD=(python PINN/run_experiment.py
               --seed=$S
               --pde=$PDE
               --opt=sgd
               --opt_params lr $LR momentum $M weight_decay $WEIGHT_DECAY
               --epochs=$EPOCHS
               --device=$PINN_DEVICE)
          CMD+=("${PDE_PARAMS[@]}")
          "${CMD[@]}"
        done
      done
    done

    # MUON
    for LR in $MUON_LRS_PINN; do
      for S in $(seq 1 $SEEDS); do
        CMD=(python PINN/run_experiment.py
             --seed=$S
             --pde=$PDE
             --opt=muon
             --opt_params lr $LR momentum $MUON_MOM weight_decay $WEIGHT_DECAY eps $EPS
             --epochs=$EPOCHS
             --device=$PINN_DEVICE)
        CMD+=("${PDE_PARAMS[@]}")
        "${CMD[@]}"
      done
    done

    # MUON_WITH_AUX_ADAM (hidden layers via MUON, others via Adam)
    for LR in $MWAA_LRS_PINN; do
      for S in $(seq 1 $SEEDS); do
        CMD=(python PINN/run_experiment.py
             --seed=$S
             --pde=$PDE
             --opt=muon_with_aux_adam
             --opt_params lr $LR momentum $MUON_MOM weight_decay $WEIGHT_DECAY betas ${BETAS_ARR[@]} eps $EPS
             --epochs=$EPOCHS
             --device=$PINN_DEVICE)
        CMD+=("${PDE_PARAMS[@]}")
        "${CMD[@]}"
      done
    done
  done
  exit 0
fi

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
      --lr=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
      --momentum=$M
  done
done

# # MUON_WITH_AUX_ADAM
for LR in $MWAA_LRS; do
  RUN_NAME=MWAA_lr${LR}_m0.95
  RUN_NAME=$RUN_NAME $TRAIN_SCRIPT \
    --ARCH=$ARCH --device=$DEVICE --dataset=$DATASET --optimizers=MUON_WITH_AUX_ADAM \
    --epochs=$EPOCHS --seeds=$SEEDS --batch_size=$BATCH \
    --lr=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
    --momentum=0.95
done


#ADAM
for OPT in ADAM ADAMW; do
  for LR in $ADAM_LRS; do
    RUN_NAME=${OPT}_lr${LR}
    RUN_NAME=$RUN_NAME $TRAIN_SCRIPT \
      --ARCH=$ARCH --device=$DEVICE --dataset=$DATASET --optimizers=$OPT \
      --epochs=$EPOCHS --seeds=$SEEDS --batch_size=$BATCH \
      --lr=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
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
      --lr=$LR --weight_decay=$WEIGHT_DECAY --betas=$BETAS --eps=$EPS \
      --momentum=$M
  done
done

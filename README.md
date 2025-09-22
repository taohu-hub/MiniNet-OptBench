# MiniNet-OptBench

A lightweight benchmark suite for training/finetuning medium-sized GPTs and comparing optimizers. This benchmark 

---

## 🚀 Installation

Set up the conda environment (includes PyTorch, NumPy, SciPy, etc.):

```bash
conda env create -f environment.yml
conda activate MiniNet-OptBench
````

---

## ⚡ Quick Start

Run the shell script to benchmark across optimizers:

```bash
bash generate_loss_table.sh
```

The command performs the following steps:

* Prepares the dataset (`data.../train.bin` and `data.../val.bin`) if not already cached.
* Runs multiple training trials across optimizers (Adam, AdamW, SGD, Muon, etc.).
* Appends results to a single CSV (`results.../bench_runs.csv`).

---

## 📝 Running Different Problems

Control the **dataset** and **optimizer sweeps** by editing environment variables at the top of `generate_loss_table.sh`.

### Example 1: Change network structure

```bash
ARCH=nanoGPT
```

Supported network structures:
* `CNN` (Convolutional Neural Nerwork for solving `CIFAR-10`)
* `nanoGPT` (with datasets for language models)


### Example 2: Change dataset

```bash
DATASET=MNIST
```

Supported datasets:

* `shakespeare`
* `shakespeare_char`
* `openwebtext`
* `openwebtext-3%` (3% sample of OpenWebText)
* `MNIST`
* `CIFAR-10`

> 💡 To add a new dataset: place `prepare.py` under `nanoGPT/data/<dataset>/` and set `DATASET=<name>`.

---

### Example 3: Run Muon with a custom learning-rate grid

```bash
MUON_LRS="0.001 0.01 0.02" 
```

---

### Example 4: Adjust epochs, seeds, and batch size

```bash
EPOCHS=50 
SEEDS="2,3,4" 
BATCH=32
```

---

## 🔧 Running Different Optimizers

Learning-rate grids are defined for each optimizer family and the learning-rate grids can be easily modified:

* **Adam / AdamW**

  ```bash
  ADAM_LRS="0.0003 0.001" 
  ADAMW_LRS="0.0003 0.001"
  ```

* **SGD**

  ```bash
  SGD_LRS="0.01 0.05" 
  ```

* **Muon**

  ```bash
  MUON_LRS="0.01 0.02" 
  ```

* **Muon + Aux Adam (hybrid)**

  ```bash
  MWAA_LRS="0.01 0.02" 
  ```

Each run is tagged with a descriptive `RUN_NAME` (e.g., `MUON_lr0.01_m0.95`) for easy curve tracking.

---

## 📊 Visualizing Results

After training, generate loss-vs-iteration plots:

```bash
python visualize_benchmark.py \
  --csv results/<DATASET>_runs.csv \
  --dataset <DATASET> \
  --split val_eval \
  --outdir results/figs
```

The plots include:

* Best run per optimizer
* All runs overlaid
* Per-optimizer figures

---

## 🙏 Acknowledgements

Thanks to **Cardinal Operations** for support and funding.

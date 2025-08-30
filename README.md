# **Benchmark for Neural Network Optimizers: Small to Medium-Scale Tasks**

## **Overview**
This repository provides a benchmark suite to evaluate neural network optimizers on small to medium-scale tasks, including:

- **NanoGPT**
- **Matrix Completion**
- **Matrix Regression**
- **CIFAR10**
- **Fashion-MNIST**

It focuses on tasks with smaller datasets and models, suitable for a wide range of research contexts.

## **Key Features**
- Optimizers evaluated on small/medium datasets like CIFAR10, Fashion-MNIST, and synthetic tasks.
- Comprehensive performance comparison: convergence rates, training efficiency, and robustness.

## **Installation**
Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git

Install dependencies:
```bash
pip install -r requirements.txt

## **Benchmarking Tasks**
- *NanoGPT*: Simplified GPT model.
- *Matrix Completion*: Recover matrix from incomplete data.
- *Matrix Regression*: Linear regression on high-dimensional matrices.
- *CIFAR10 & Fashion-MNIST*: Image classification tasks.

## **Usage**
Run the benchark:

```bash
python run_benchmark.py

Or specify a task:
```bash
python run_benchmark.py --task cifar10

## **Contributing**
1. Fork the repository.
2. Create a future branch.
3. Submit a pull request.

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

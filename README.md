# Sisyphus: A Cautionary Tale of Using Low-Degree Polynomial Activations in Privacy-Preserving Deep Learning

This repository contains the code for the Sisyphus framework, a set of methods for wholesale ReLU replacement using polynomial activation functions in Private Inference. The repo is structured as followed:

1. `models`: PyTorch implementation of various network architectures
2. `data`: Instructions for downloading MNIST, CIFAR, and TinyImageNet
2. `experiments`
    - `baselines`: pipeline to train baseline networks with ReLU
    - `tayloy_approx`: Taylor series approximation of ReLU
    - `poly_regression`: Polynomial regression fit of ReLU
    - `quail`: **Qua**dratic **I**mitation **L**earning training pipeline
    - `approxminmax_quail`: ApproxMinMaxNorm implementation
    - `test_networks`: simply test loss and accuracy evaluation script

## Installation
Clone this repo:
```
git clone https://github.com/sisyphus-project/sisyphus-ppml.git
cd sisyphus-ppml
```
Install the required Python packages:
```
pip install -r requirements.txt
```
Setup two environment variables (for the datasets and models). You may want to add these environment variables to your `bashrc` file.
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/models"
export DATASET_DIR=$(pwd)/data
```
Follow the instructions in the `data` directory to download the datasets. We use [wandb](https://docs.wandb.ai/) to log our experiments. 

## Example
To run a baseline model, move to the `baselines` directory and run:
```
python train_mnist.py --project=sisyphus-baseline --name=mnist-mlp --model=mlp_bn
```
For more detailed instructions on running experiments, please refer to the READMEs in each subdirectory.

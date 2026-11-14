#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/require_published.py
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -y -n neuralplane5090 python=3.10 pip
conda activate neuralplane5090
export CUDA_HOME="${CUDA_HOME:-$HOME/reconstr/.cuda/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export TCNN_CUDA_ARCHITECTURES=120
export MAX_JOBS=8
export PIP_CONSTRAINT="$PWD/scripts/constraints5090.txt"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install pip==25.1.1 setuptools==75.8.0 wheel==0.45.1 ninja==1.11.1.4
git submodule update --init --recursive
python -m pip install -e .
python -m pip install --no-build-isolation thirdparty/tiny-cuda-nn/bindings/torch
python -m pip install -e thirdparty/segment-anything -e thirdparty/Hierarchical-Localization
python -m pip install gdown==5.2.0
python -m neuralplane.scripts.download_models --dst checkpoints
mkdir -p environment
python -m pip freeze > environment/pip-freeze.txt
conda list --explicit > environment/conda-explicit.txt
sha256sum checkpoints/* > environment/checkpoints.sha256
nvcc --version > environment/nvcc.txt
nvidia-smi -q > environment/nvidia-smi.txt
echo "Run scripts/check_runtime.py after acquiring the campaign GPU slot."

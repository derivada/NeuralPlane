#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/require_published.py
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -y -n neuralplane5090 python=3.10 pip
conda activate neuralplane5090
# every install below must land in this env, never in whichever env the calling shell had activated
[ "$CONDA_PREFIX" = "$HOME/miniconda3/envs/neuralplane5090" ] || { echo "wrong conda env: $CONDA_PREFIX"; exit 1; }
ENVPY="$CONDA_PREFIX/bin/python"
export CUDA_HOME="${CUDA_HOME:-$HOME/reconstr/.cuda/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export TCNN_CUDA_ARCHITECTURES=120
export MAX_JOBS=8
export PIP_CONSTRAINT="$PWD/scripts/constraints5090.txt"
"$ENVPY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$ENVPY" -m pip install pip==25.1.1 setuptools==75.8.0 wheel==0.45.1 ninja==1.11.1.4
git submodule update --init --recursive
"$ENVPY" -m pip install -e .
"$ENVPY" -m pip install --no-build-isolation thirdparty/tiny-cuda-nn/bindings/torch
"$ENVPY" -m pip install -e thirdparty/segment-anything -e thirdparty/Hierarchical-Localization
"$ENVPY" -m pip install gdown==5.2.0 pycocotools
"$ENVPY" -m neuralplane.scripts.download_models --dst checkpoints
mkdir -p environment
"$ENVPY" -m pip freeze > environment/pip-freeze.txt
conda list --explicit > environment/conda-explicit.txt
sha256sum checkpoints/* > environment/checkpoints.sha256
nvcc --version > environment/nvcc.txt
nvidia-smi -q > environment/nvidia-smi.txt
echo "Run scripts/check_runtime.py after acquiring the campaign GPU slot."

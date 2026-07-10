#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-2dKeypointHand}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_PARENT="$(dirname "$REPO_ROOT")"
DEPS_DIR="${DEPS_DIR:-$REPO_PARENT}"
MMCV_DIR="${DEPS_DIR}/mmcv"

echo "Repository root: ${REPO_ROOT}"
echo "Environment name: ${ENV_NAME}"

cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available on PATH."
  echo "Install Miniconda or Anaconda first, then rerun this script."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists."
  echo "Skipping environment creation."
else
  echo "Creating conda environment from environment.yml..."
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

conda activate "${ENV_NAME}"

echo
echo "Installing MMCV v1.3.9 from source..."

git clone https://github.com/open-mmlab/mmcv.git "$MMCV_DIR"

cd "${MMCV_DIR}"
git checkout v1.3.9

MMCV_WITH_OPS=1 pip install -e .

echo
echo "Installing local ViTPose/MMPose repository..."

cd "${REPO_ROOT}"
pip install -v -e .
pip install timm==0.4.9 einops

echo
echo "Checking package consistency..."
python -m pip check

echo
echo "Verifying core imports..."

python - <<'PY'
import sys
print("Python:", sys.version)

import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import torchvision
print("torchvision:", torchvision.__version__)

import mmcv
print("mmcv:", mmcv.__version__)

import mmpose
print("mmpose:", mmpose.__version__)
print("mmpose path:", mmpose.__file__)

import timm
print("timm:", timm.__version__)

import einops
print("einops:", einops.__version__)
PY

echo
echo "Environment setup complete."
echo
echo "To use this environment later, run:"
echo "  conda activate ${ENV_NAME}"
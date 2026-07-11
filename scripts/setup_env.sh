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

if ! ldconfig -p 2>/dev/null | grep -q "libGL.so.1"; then
  echo "ERROR: libGL.so.1 is missing."
  echo
  echo "OpenCV needs this system library. On Ubuntu/Debian, install it with:"
  echo "  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"
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
echo "Installing MMCV v1.3.9 prebuilt wheel..."

python -m pip install mmcv-full==1.3.9 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

echo
echo "Installing local ViTPose/MMPose repository..."

cd "${REPO_ROOT}"
python -m pip install -v -e .
python -m pip install timm==0.4.9 einops==0.8.1

echo
echo "Registering vendored third-party packages..."

python - <<'PY'
import site
from pathlib import Path

repo_root = Path.cwd()
site_packages = Path(site.getsitepackages()[0])
pth_file = site_packages / "thesis_third_party.pth"

third_party_paths = [
    repo_root / "third_party" / "dinov2",
    repo_root / "third_party" / "dinov3",
]

existing = [str(p.resolve()) for p in third_party_paths if p.exists()]

if not existing:
    raise RuntimeError(
        "No vendored third-party packages found. Expected one or both of: "
        "third_party/dinov2, third_party/dinov3"
    )

pth_file.write_text("\n".join(existing) + "\n")

print(f"Wrote {pth_file}")
for path in existing:
    print(f"Added to Python path: {path}")
PY

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

from mmpose.models import build_posenet   # exercises model registry
x = (torch.randn(8, device="cuda") * 2).sum()  # forces a real CUDA kernel
print("CUDA compute OK:", float(x) == float(x))
PY

echo
echo "Environment setup complete."
echo
echo "To use this environment later, run:"
echo "  conda activate ${ENV_NAME}"
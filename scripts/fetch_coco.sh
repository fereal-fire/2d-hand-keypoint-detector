#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
COCO_ROOT="${DATA_ROOT}/coco"
ARCHIVE_DIR="${COCO_ROOT}/raw_archives"
ANN_DIR="${COCO_ROOT}/annotations"

mkdir -p "${ARCHIVE_DIR}" "${ANN_DIR}"
echo "COCO data root:    ${COCO_ROOT}"
echo "Archive directory: ${ARCHIVE_DIR}"

cd "${ARCHIVE_DIR}"

echo
echo "Downloading training data..."
wget -c "http://images.cocodataset.org/zips/train2017.zip"

echo
echo "Extracting training data..."
unzip -q train2017.zip -d "${COCO_ROOT}"

echo
echo "Downloading validation data..."
wget -c "http://images.cocodataset.org/zips/val2017.zip"

echo
echo "Extracting validation data..."
unzip -q val2017.zip -d "${COCO_ROOT}"

echo
echo "Downloading train annotations..."
gdown https://drive.google.com/uc?id=1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf -O "${ANN_DIR}/coco_wholebody_train_v1.0.json"

head -c 1 "${ANN_DIR}/coco_wholebody_train_v1.0.json" | grep -q '{' || { echo "ERROR: not JSON (Drive quota?)"; exit 1; }

echo
echo "Downloading validation annotations..."
gdown https://drive.google.com/uc?id=1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb -O "${ANN_DIR}/coco_wholebody_val_v1.0.json"

head -c 1 "${ANN_DIR}/coco_wholebody_val_v1.0.json" | grep -q '{' || { echo "ERROR: not JSON (Drive quota?)"; exit 1; }
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
HAMER_ROOT="${DATA_ROOT}/hamer"
mkdir -p "${HAMER_ROOT}"

DATASET_TARS_ROOT="${DATASET_TARS_ROOT:-${HAMER_ROOT}/hamer_training_data/dataset_tars}" # If data downloaded to another repository, can overwrite this variable to point to the correct location.

if [[ ! -d "${DATASET_TARS_ROOT}" ]]; then
    echo "ERROR: Dataset tar directory not found:"
    echo "  ${DATASET_TARS_ROOT}"
    exit 1
fi

echo
echo "Extracting nested dataset tar shards..."
echo "Nested tar root: ${DATASET_TARS_ROOT}"
echo "Nested output root: ${HAMER_ROOT}"

REMOVE="${REMOVE:-0}"

find "${DATASET_TARS_ROOT}" -type f -name "*.tar" -print0 |
while IFS= read -r -d '' tar_file; do
  echo "Extracting ${tar_file}"
  tar --warning=no-unknown-keyword --exclude=".*" -xf "${tar_file}" -C "${HAMER_ROOT}" 
  if [[ "${REMOVE}" == "1" ]]; then
    echo "Removing ${tar_file}"
    rm -v "${tar_file}"
  fi
done
echo
echo "HaMeR data extracted."
echo "Data root: ${HAMER_ROOT}"
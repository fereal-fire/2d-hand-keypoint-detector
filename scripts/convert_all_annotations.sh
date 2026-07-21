#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
HAMER_ROOT="${DATA_ROOT}/hamer"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${REPO_ROOT}/scripts/convert_hamer.py}"

PAD_SCALE="${PAD_SCALE:-1.0}"
WORKERS="${WORKERS:-$(nproc)}"
REORDER="${REORDER:-}"     # 21-index comma-separated mapping, empty = no reorder
FORCE="${FORCE:-0}"        # FORCE=1 re-converts even if output exists

# Per-dataset --size cap (max annotations to convert), two ways to set it:
#   SIZES_FILE=path/to/counts.csv   CSV from scripts/count_annotations.py
#                                    (columns: group,subset,n_images,n_annotations,status,path)
#                                    -> uses n_annotations for group=hamer, status=ok rows
#   SIZES="name=size,name=size,..." inline override, e.g. SIZES="rhd-train=6566,mtc-train=13133"
#   SIZES entries win over SIZES_FILE entries for the same dataset name.
# A dataset with no matching entry is converted in full (--size -1), same as before.
SIZES_FILE="${SIZES_FILE:-}"
SIZES="${SIZES:-}"

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: converter not found: ${CONVERT_SCRIPT}"
  exit 1
fi

if [[ ! -d "${HAMER_ROOT}" ]]; then
  echo "ERROR: HaMeR data root not found: ${HAMER_ROOT}"
  exit 1
fi

declare -A DATASET_SIZE=()

if [[ -n "${SIZES_FILE}" ]]; then
  if [[ ! -f "${SIZES_FILE}" ]]; then
    echo "ERROR: SIZES_FILE not found: ${SIZES_FILE}"
    exit 1
  fi
  # CSV: group,subset,n_images,n_annotations,status,path -- skip header, only hamer/ok rows
  while IFS=, read -r group subset n_images n_annotations status path; do
    [[ "${group}" == "group" ]] && continue   # header
    [[ "${group}" == "hamer" && "${status}" == "ok" && -n "${n_annotations}" ]] || continue
    DATASET_SIZE["${subset}"]="${n_annotations}"
  done < "${SIZES_FILE}"
fi

if [[ -n "${SIZES}" ]]; then
  IFS=',' read -ra _pairs <<< "${SIZES}"
  for pair in "${_pairs[@]}"; do
    key="${pair%%=*}"
    val="${pair#*=}"
    [[ -n "${key}" && -n "${val}" ]] || continue
    DATASET_SIZE["${key}"]="${val}"
  done
fi

echo "HaMeR data root: ${HAMER_ROOT}"
echo "Converter:       ${CONVERT_SCRIPT}"
echo "Pad scale:       ${PAD_SCALE}"
echo "Reorder:         ${REORDER:-<none>}"
if [[ ${#DATASET_SIZE[@]} -gt 0 ]]; then
  echo "Size caps:"
  for k in "${!DATASET_SIZE[@]}"; do
    echo "  ${k} -> ${DATASET_SIZE[$k]}"
  done
fi
echo

REORDER_ARGS=()
if [[ -n "${REORDER}" ]]; then
  REORDER_ARGS=(--reorder "${REORDER}")
fi

converted=0
skipped=0
failed=()

for dataset_dir in "${HAMER_ROOT}"/*/; do
  dataset_dir="${dataset_dir%/}"
  name="$(basename "${dataset_dir}")"

  # Skip non-dataset directories
  case "${name}" in
    raw_archives|hamer_training_data|annotations) continue ;;
  esac

  # Only convert directories that actually contain HaMeR metadata files
  if [[ -z "$(find "${dataset_dir}" -type f -name '*.data.pyd' -print -quit)" ]]; then
    echo "SKIP  ${name}: no *.data.pyd files"
    continue
  fi

  out_dir="${dataset_dir}/annotations"
  out_json="${out_dir}/coco_annotations.json"

  if [[ -f "${out_json}" && "${FORCE}" != "1" ]]; then
    echo "SKIP  ${name}: ${out_json} already exists (FORCE=1 to redo)"
    skipped=$((skipped + 1))
    continue
  fi

  SIZE_ARGS=()
  if [[ -n "${DATASET_SIZE[${name}]+set}" ]]; then
    SIZE_ARGS=(--size "${DATASET_SIZE[${name}]}")
    echo "CONVERT ${name} (--size ${DATASET_SIZE[${name}]})"
  else
    echo "CONVERT ${name}"
  fi
  if python "${CONVERT_SCRIPT}" \
      --input-dir "${dataset_dir}" \
      --output-dir "${out_dir}" \
      --description "HaMeR ${name} converted to COCO wholebody-hand format" \
      --pad-scale "${PAD_SCALE}" \
      --workers "${WORKERS}" \
      "${SIZE_ARGS[@]}" \
      "${REORDER_ARGS[@]}"; then
    converted=$((converted + 1))
  else
    echo "ERROR: conversion failed for ${name}"
    failed+=("${name}")
  fi
  echo
done

echo "Done. Converted: ${converted}  Skipped: ${skipped}  Failed: ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'Failed datasets:\n'
  printf '  %s\n' "${failed[@]}"
  exit 1
fi

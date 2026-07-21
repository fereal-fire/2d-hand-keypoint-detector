#!/usr/bin/env bash
# Run eval_composed.py for one model over all HInt test configurations.
# Usage:
#   bash scripts/run_hint_eval_all.sh <model_config.py> <checkpoint.pth>
# Env overrides:
#   TEST_CONFIG_DIR  dir containing the per-split test configs
#   NAMES            datasets  (default: "newdays epick"; add ego4d if frames present)
#   SPLITS           splits    (default: "all vis occ")
#   DATA_ROOT        data dir  (default: data)
set -euo pipefail

MODEL_CONFIG=${1:?usage: run_hint_eval_all.sh <model_config.py> <checkpoint.pth>}
CKPT=${2:?usage: run_hint_eval_all.sh <model_config.py> <checkpoint.pth>}

TEST_CONFIG_DIR=${TEST_CONFIG_DIR:-configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset/hint_tests}
DATA_ROOT=${DATA_ROOT:-data}
NAMES=${NAMES:-"newdays epick"}
SPLITS=${SPLITS:-"all vis occ"}

RUN=$(basename "$CKPT" .pth)
LOG_DIR="eval_logs/$(basename "$MODEL_CONFIG" .py)_${RUN}"
mkdir -p "$LOG_DIR"

failures=()

for name in $NAMES; do
  for split in $SPLITS; do
    tag="${name}_${split}"
    test_cfg="$TEST_CONFIG_DIR/hint_${name}_${split}.py"
    npz="$DATA_ROOT/hamer_evaluation_data/TEST_${name}_img_${split}.npz"
    preds="model_predictions/${tag}/${RUN}.pkl"

    if [[ ! -f "$test_cfg" ]]; then
      echo "[SKIP] $tag: no test config at $test_cfg"
      failures+=("$tag")
      continue
    fi
    if [[ ! -f "$npz" ]]; then
      echo "[SKIP] $tag: no GT npz at $npz"
      failures+=("$tag")
      continue
    fi

    echo
    echo "===== $tag ====="
    if python scripts/eval_composed.py \
        "$MODEL_CONFIG" "$CKPT" "$test_cfg" \
        --npz "$npz" \
        --out "$preds" 2>&1 | tee "$LOG_DIR/${tag}.log"; then
      :
    else
      echo "[FAIL] $tag (see $LOG_DIR/${tag}.log)"
      failures+=("$tag")
    fi
  done
done

echo
echo "===== PCK summary ====="
grep -h -A 4 "=== HaMeR PCK" "$LOG_DIR"/*.log || echo "(no results parsed)"

if [[ ${#failures[@]} -gt 0 ]]; then
  echo
  echo "Failed/skipped: ${failures[*]}"
  exit 1
fi
echo
echo "All evaluations complete. Logs in $LOG_DIR/"
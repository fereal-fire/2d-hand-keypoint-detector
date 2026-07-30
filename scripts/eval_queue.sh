#!/usr/bin/env bash
# Queue several eval_pck.py runs (different configs/checkpoints) back to back.
#
# 1) Edit the RUNS list below: one "config checkpoint" pair per line.
# 2) Launch detached so it keeps running after you log out:
#      nohup bash jobs/eval_queue.sh > eval_queue.log 2>&1 &
#      tail -f eval_queue.log            # watch progress
#    (tmux alternative: tmux new -d -s eval 'bash jobs/eval_queue.sh |& tee eval_queue.log')
#
# Each run gets its own log in eval_logs/ and its own --pred-dir, named after
# <work_dir>_<ckpt>, so two checkpoints both called latest.pth can't overwrite
# each other's prediction pkls. A failed run is reported and the queue moves on.
set -u -o pipefail

RUNS=(
  "configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset/DINOv3_base_hand_multidataset.py work_dirs/DINOv3_like_for_like/latest.pth"
  # "configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset/MAE_base_hand_multidataset.py work_dirs/MAE_like_for_like/latest.pth"
  # "configs/.../DINOv3_base_hand_multidataset.py work_dirs/DINOv3_like_for_like/epoch_20.pth"
)

LOG_DIR=eval_logs
mkdir -p "$LOG_DIR"
overall=0

for run in "${RUNS[@]}"; do
  read -r cfg ckpt <<<"$run"
  name="$(basename "$(dirname "$ckpt")")_$(basename "$ckpt" .pth)"
  log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START $name"
  python scripts/eval_pck.py "$cfg" "$ckpt" --pred-dir "model_predictions/$name" >"$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[$(date '+%F %T')] DONE  $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (exit $rc, see $log)"
    overall=1
  fi
done

echo
echo "================ combined summaries ================"
grep -H -A8 '^===== Summary' "$LOG_DIR"/*.log || echo "(no summaries found)"
exit "$overall"
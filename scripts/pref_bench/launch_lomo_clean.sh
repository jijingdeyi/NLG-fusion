#!/bin/bash
# Launch clean LOMO + matched-size controls on free GPUs.
set -uo pipefail
unset CUDA_VISIBLE_DEVICES
CFG=/data/ykx/pref_bench/config.yaml
ROOT=/home/ykx/NLG-fusion
cd "$ROOT"
mkdir -p logs/ablations

pkill -f -- '--tag lomo_method_' 2>/dev/null || true
pkill -f -- '--tag matched_method_' 2>/dev/null || true
sleep 2

run() {
  local gpu="$1"; shift
  local log="$1"; shift
  echo "[$(date +%H:%M:%S)] launch gpu=$gpu -> logs/ablations/${log}.log -- $*"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/pref_bench/train_scorer_stage1.py \
    --config "$CFG" --gpu 0 --num-workers 2 "$@" \
    > "logs/ablations/${log}.log" 2>&1 &
  echo "  pid=$!"
}

run 0 lomo2_a --holdout-method method_a --tag lomo_method_a
run 1 lomo2_b --holdout-method method_b --tag lomo_method_b
run 2 lomo2_c --holdout-method method_c --tag lomo_method_c
run 4 lomo2_d --holdout-method method_d --tag lomo_method_d
run 5 matched_a --matched-train-size 423 --tag matched_method_a
run 6 matched_b --matched-train-size 872 --tag matched_method_b
run 7 matched_c --matched-train-size 876 --tag matched_method_c

echo "[$(date +%H:%M:%S)] waiting 90s for VRAM to allocate, then poll for matched_d ..."
sleep 90
echo "[$(date +%H:%M:%S)] waiting for free GPU for matched_d ..."
while true; do
  if pgrep -f -- '--tag matched_method_d' >/dev/null 2>&1; then
    break
  fi
  if grep -q '^best:' logs/ablations/matched_d.log 2>/dev/null; then
    break
  fi
  for g in 0 5 1 2 4 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [[ "${used:-99999}" -lt 400 ]]; then
      echo "[$(date +%H:%M:%S)] GPU $g free (${used}MiB); launching matched_d"
      run "$g" matched_d --matched-train-size 869 --tag matched_method_d
      break 2
    fi
  done
  sleep 20
done

# Wait until all 8 logs show a final "best:" line
echo "[$(date +%H:%M:%S)] waiting for all trainings to finish ..."
while true; do
  n=0
  for f in lomo2_a lomo2_b lomo2_c lomo2_d matched_a matched_b matched_c matched_d; do
    if grep -q '^best:' "logs/ablations/${f}.log" 2>/dev/null; then
      n=$((n+1))
    fi
  done
  echo "[$(date +%H:%M:%S)] finished $n/8"
  if [[ "$n" -ge 8 ]]; then
    break
  fi
  sleep 60
done
echo "[$(date +%H:%M:%S)] all done"

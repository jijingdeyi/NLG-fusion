#!/usr/bin/env bash
# SRC modulation ablation (Table tab:src-ablation).
# Train global_film and residual_film; SRC row reuses full model (control_plus / tab:llvip Ours auto).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-/data/ykx/LLVIP/train}"
CKPT="${CKPT:-checkpoint}"

run_one() {
  local mode="$1"
  local run_name="src_abl_${mode}"
  echo "========== ${run_name} (src_mode=${mode}) =========="
  python scripts/train.py \
    --data "${DATA}" \
    --ckpt "${CKPT}" \
    --run_name "${run_name}" \
    --ablation_variant full \
    --src_mode "${mode}"
}

for m in global_film residual_film; do
  run_one "${m}"
done

echo "Done. Evaluate on LLVIP 40-pair test:"
echo "  python scripts/pred.py --ckpt checkpoint/src_abl_global_film/...best.ckpt --data <llvip_test> --dst <out>"
echo "  python scripts/pred.py --ckpt checkpoint/src_abl_residual_film/...best.ckpt --data <llvip_test> --dst <out>"
echo "SRC (proposed) row: reuse full checkpoint (src_mode=src), e.g. control_plus."

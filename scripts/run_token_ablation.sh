#!/usr/bin/env bash
# Token-ablation training (Table tab:token-ablation). Row (5) = default full model (control_plus already trained).
# Row (2) online CLIP: compare Text-IF in tab:llvip — not trained here.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-/data/ykx/LLVIP/train}"
CKPT="${CKPT:-checkpoint}"

run_one() {
  local variant="$1"
  local run_name="tok_abl_${variant}"
  echo "========== ${run_name} (paper ${variant}) =========="
  python scripts/train.py \
    --data "${DATA}" \
    --ckpt "${CKPT}" \
    --run_name "${run_name}" \
    --ablation_variant "${variant}"
}

# Rows (1)(3)(4)(6) — each gets its own subfolder under checkpoint/
for v in no_token freeze_token no_router random_token; do
  run_one "${v}"
done

echo "Done. Evaluate each checkpoint with:"
echo "  python scripts/pred.py --ckpt checkpoint/tok_abl_<variant>/tok_abl_<variant>-best-eXXX.ckpt"
echo "Row (5) reuses control_plus / tab:llvip Ours (auto). Row (2): use Text-IF in tab:llvip."

#!/usr/bin/env bash
# Compute a FID-vs-budget curve for one centralized run's checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR=$(ls -dt logs/central/*tra-budget-test* | head -1)
echo "run: $RUN_DIR"
for CKPT in "$RUN_DIR"/checkpoints/global_round_*.pt; do
  ROUND=$(basename "$CKPT" | sed 's/[^0-9]//g')
  OUT="$RUN_DIR/generated/round_$ROUND"
  if [[ -f "$OUT/fid.json" ]]; then
    echo "round $ROUND: cached"
    continue
  fi
  ./.venv/bin/python BFL/evaluate_ddpm.py "$CKPT" --output-dir "$OUT" \
    --num-samples 2000 --batch-size 64 --device cuda --seed 9100 >/dev/null 2>&1
  ./.venv/bin/python BFL/evaluate_fid.py "$OUT/samples.pt" --output "$OUT/fid.json" \
    --max-real-samples 10000 --batch-size 64 --device cuda >/dev/null 2>&1
  python3 -c "import json,sys; d=json.load(open('$OUT/fid.json')); print('round $ROUND FID', round(d['fid'],1))"
done

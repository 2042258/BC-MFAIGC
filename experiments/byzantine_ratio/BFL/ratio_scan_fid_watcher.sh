#!/usr/bin/env bash
# Wait for the ratio-scan training process to exit, then compute FID for all 36 runs.
cd PATH_REQUIRED_22
TRAIN_PID="$1"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[watcher] training pid $TRAIN_PID exited at $(date)" >> BFL/ratio_scan_fid_v1.log
.venv/bin/python BFL/run_fid.py \
  --results BFL/ratio_scan_results_v1.jsonl \
  --output BFL/ratio_scan_fid_v1.jsonl \
  --num-samples 2000 --max-real-samples 10000 \
  >> BFL/ratio_scan_fid_v1.log 2>&1
echo "[watcher] FID done at $(date)" >> BFL/ratio_scan_fid_v1.log

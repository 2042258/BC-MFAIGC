#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
P=phase1_validation
OUT="$P/live_status.log"
while pgrep -f '[r]un_formal_pipeline.sh' >/dev/null 2>&1; do
  now=$(date '+%F %T')
  stage='waiting'
  if [[ -f "$P/matrix_core_100r_v4.json" ]]; then
    stage="core_matrix $(wc -l < "$P/results_core_100r_v4.jsonl" 2>/dev/null || echo 0)/60"
  elif [[ -f "$P/matrix_sweep_100r_v4.json" ]]; then
    stage="sigma_sweep $(wc -l < "$P/results_sweep_100r_v4.jsonl" 2>/dev/null || echo 0)/15"
  elif [[ -f "$P/results_calib100_v4.jsonl" ]]; then
    stage="calibration $(wc -l < "$P/results_calib100_v4.jsonl")/10"
  fi
  jobs=$(pgrep -fc '[B]FL/main.py' || true)
  printf '%s stage=%s jobs=%s\n' "$now" "$stage" "$jobs" >> "$OUT"
  sleep 30
done
printf '%s pipeline=stopped_or_finished\n' "$(date '+%F %T')" >> "$OUT"

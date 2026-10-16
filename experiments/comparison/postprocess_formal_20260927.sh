#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY="./.venv/bin/python"
P="phase1_validation"
LOG="$P/postprocess_formal_20260927.log"
exec > >(tee -a "$LOG") 2>&1

echo "$(date '+%F %T') postprocess supervisor started"
CENTRAL_RECORDS="logs/central/20260927_111946_reference-central-clean-93000/central_training_records.jsonl"
while true; do
  n=0
  [[ -f "$CENTRAL_RECORDS" ]] && n=$(wc -l < "$CENTRAL_RECORDS")
  running=0
  ps -u "$(id -un)" -o cmd= | grep -E 'train_central_ddpm.py.*reference-central-clean-93000' | grep -v grep >/dev/null && running=1 || true
  echo "$(date '+%F %T') central records: $n/100 running=$running"
  if [[ "$n" -ge 100 && "$running" -eq 0 ]]; then break; fi
  if [[ "$running" -eq 0 && "$n" -lt 100 ]]; then echo "central reference stopped before 100 epochs"; exit 2; fi
  sleep 30
done

"$PY" BFL/collect_central.py \
  --manifest "$P/formal_experiment_manifest_reference_v1_released_20260927.json" \
  --output "$P/formal_reference_results_20260927.jsonl"

"$PY" collect_formal_runs_20260927.py \
  --logs-dir logs/blogs \
  --manifests "$P/formal_experiment_manifest_core_v1_released_20260926.json" "$P/formal_experiment_manifest_ablation_v1_released_20260927.json" \
  --output "$P/formal_federated_collected_20260927.jsonl"

"$PY" BFL/run_fid.py \
  --results "$P/formal_core_results_20260926.jsonl" "$P/formal_ablation_results_20260927.jsonl" "$P/formal_reference_results_20260927.jsonl" \
  --output "$P/formal_fid_20260927.jsonl" \
  --num-samples 5000 --max-real-samples 10000 --batch-size 64 --device cuda

"$PY" BFL/analyze_matrix.py \
  --runs "$P/formal_federated_collected_20260927.jsonl" \
  --output "$P/formal_federated_summary_20260927.json"

"$PY" plot_formal_results_20260927.py \
  --runs "$P/formal_federated_collected_20260927.jsonl" \
  --fid "$P/formal_fid_20260927.jsonl" \
  --output-dir "$P/formal_plots_20260927"

date '+%F %T' > "$P/formal_postprocess_complete_20260927.txt"
echo "$(date '+%F %T') postprocess complete"

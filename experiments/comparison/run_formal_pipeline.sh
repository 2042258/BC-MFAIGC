#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="./.venv/bin/python"
P=phase1_validation
mkdir -p "$P"

wait_lines() {
  local file=$1 expected=$2
  while true; do
    local n=0
    [[ -f "$file" ]] && n=$(wc -l < "$file")
    echo "$(date '+%F %T') wait $file: $n/$expected"
    [[ "$n" -ge "$expected" ]] && break
    sleep 60
  done
}

# A: formal 100-round clean calibration. The matrix workers are launched
# separately; this driver waits for all ten completed records.
wait_lines "$P/results_calib100_v4.jsonl" 10
$PYTHON BFL/collect_runs.py --logs-dir logs/blogs \
  --manifest "$P/matrix_calib100_v4.json" \
  --output "$P/runs_calib100_v4.jsonl"
$PYTHON BFL/analyze_threshold.py --runs "$P/runs_calib100_v4.jsonl" \
  --output "$P/frozen_threshold_100r_v4.json"
THRESHOLD=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$P/frozen_threshold_100r_v4.json")
echo "Frozen formal threshold: $THRESHOLD"

# B: threshold-near sigma sweep.
$PYTHON BFL/make_matrix.py sweep --rounds 100 --seeds 3 --seed-start 21000 \
  --threshold "$THRESHOLD" --noise-scales 0.0025 0.005 0.01 0.02 0.05 \
  --output "$P/matrix_sweep_100r_v4.json"
$PYTHON BFL/run_matrix.py --manifest "$P/matrix_sweep_100r_v4.json" \
  --workers 3 --results "$P/results_sweep_100r_v4.jsonl" --timeout-seconds 7200
$PYTHON BFL/collect_runs.py --logs-dir logs/blogs \
  --manifest "$P/matrix_sweep_100r_v4.json" \
  --output "$P/runs_sweep_100r_v4.jsonl"
$PYTHON BFL/choose_sigma.py --runs "$P/runs_sweep_100r_v4.jsonl" \
  --output "$P/frozen_sigma_100r_v4.json"
SIGMA=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["sigma"])' "$P/frozen_sigma_100r_v4.json")
echo "Frozen near-threshold sigma: $SIGMA"

# C: core matrix (three required methods, clean + 10/30/60%, five seeds).
$PYTHON BFL/make_matrix.py matrix --rounds 100 --seeds 5 --seed-start 30000 \
  --threshold "$THRESHOLD" --noise-scale "$SIGMA" \
  --output "$P/matrix_core_100r_v4.json"
$PYTHON BFL/run_matrix.py --manifest "$P/matrix_core_100r_v4.json" \
  --workers 3 --results "$P/results_core_100r_v4.jsonl" --timeout-seconds 7200
$PYTHON BFL/collect_runs.py --logs-dir logs/blogs \
  --manifest "$P/matrix_core_100r_v4.json" \
  --output "$P/runs_core_100r_v4.jsonl"
$PYTHON BFL/analyze_matrix.py --runs "$P/runs_core_100r_v4.jsonl" \
  --output "$P/summary_core_100r_v4.json"

# D: centralized DDPM reference and contaminated TRA controls.
CENTRAL_EPOCHS=$($PYTHON - <<'PY'
import json, math
rows=[json.loads(x) for x in open('phase1_validation/runs_calib100_v4.jsonl')]
values=[r.get('worker_training_samples',0) for r in rows if r.get('worker_training_samples',0)>0]
print(max(1, math.ceil(sum(values)/len(values)/50000)))
PY
)
echo "Budget-matched central epochs: $CENTRAL_EPOCHS"
$PYTHON BFL/make_matrix.py tra --epochs "$CENTRAL_EPOCHS" --seeds 5 \
  --seed-start 40000 --noise-scale "$SIGMA" --output "$P/matrix_tra_v4.json"
$PYTHON BFL/run_central_matrix.py --manifest "$P/matrix_tra_v4.json" \
  --workers 3 --results "$P/results_tra_v4.jsonl" --timeout-seconds 7200
$PYTHON BFL/collect_central.py --manifest "$P/matrix_tra_v4.json" \
  --output "$P/runs_tra_v4.jsonl"

# E: uptime and non-IID-local-test stress runs.
$PYTHON BFL/make_matrix.py uptime --rounds 100 --seeds 3 --seed-start 50000 \
  --threshold "$THRESHOLD" --noise-scale "$SIGMA" --output "$P/matrix_uptime_v4.json"
$PYTHON BFL/run_matrix.py --manifest "$P/matrix_uptime_v4.json" \
  --workers 3 --results "$P/results_uptime_v4.jsonl" --timeout-seconds 7200
$PYTHON BFL/collect_runs.py --logs-dir logs/blogs --manifest "$P/matrix_uptime_v4.json" \
  --output "$P/runs_uptime_v4.jsonl"

$PYTHON BFL/make_matrix.py noniid_validation --rounds 100 --seeds 3 --seed-start 55000 \
  --threshold "$THRESHOLD" --noise-scale "$SIGMA" --output "$P/matrix_noniid_validation_v4.json"
$PYTHON BFL/run_matrix.py --manifest "$P/matrix_noniid_validation_v4.json" \
  --workers 3 --results "$P/results_noniid_validation_v4.jsonl" --timeout-seconds 7200
$PYTHON BFL/collect_runs.py --logs-dir logs/blogs --manifest "$P/matrix_noniid_validation_v4.json" \
  --output "$P/runs_noniid_validation_v4.jsonl"

# G1: dedicated single-tenant timing sample (three required model paths).
$PYTHON BFL/make_matrix.py matrix --rounds 100 --seeds 1 --seed-start 60000 \
  --malicious-nodes 0 --threshold "$THRESHOLD" --noise-scale "$SIGMA" \
  --output "$P/matrix_timing_v4.json"
$PYTHON BFL/run_matrix.py --manifest "$P/matrix_timing_v4.json" \
  --workers 1 --results "$P/results_timing_v4.jsonl" --timeout-seconds 7200
$PYTHON BFL/collect_runs.py --logs-dir logs/blogs --manifest "$P/matrix_timing_v4.json" \
  --output "$P/runs_timing_v4.jsonl"

# G2: all planned metric summaries and FID. This is intentionally the final
# expensive step because every final checkpoint must exist first.
$PYTHON BFL/analyze_matrix.py \
  --runs "$P/runs_core_100r_v4.jsonl" "$P/runs_uptime_v4.jsonl" "$P/runs_noniid_validation_v4.jsonl" "$P/runs_timing_v4.jsonl" \
  --output "$P/summary_all_federated_v4.json"
$PYTHON BFL/run_fid.py \
  --results "$P/runs_core_100r_v4.jsonl" "$P/runs_tra_v4.jsonl" "$P/runs_uptime_v4.jsonl" "$P/runs_noniid_validation_v4.jsonl" \
  --output "$P/fid_all_v4.jsonl" --num-samples 2000 --max-real-samples 10000 --batch-size 64 --device cuda

echo "Formal experiment pipeline complete: training, TRA, uptime, non-IID, timing and FID." 

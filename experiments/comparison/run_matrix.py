"""Execute a subset of a run manifest with bounded parallelism."""

from __future__ import annotations

import os
import sys
import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
from pathlib import Path

BFL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BFL_DIR.parent
VENV_PYTHON = Path(os.environ.get("BFL_PYTHON", sys.executable))
LOGS_DIR = PROJECT_ROOT / "logs" / "blogs"

RESULTS_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_run(run_dir: Path, run_spec: dict, wall_seconds: float) -> dict:
    validations = read_jsonl(run_dir / "validation_records.jsonl")
    decisions = read_jsonl(run_dir / "aggregation_decisions.jsonl")
    timings = read_jsonl(run_dir / "timing_records.jsonl")
    malicious_scores = [r["score"] for r in validations if r["worker_malicious"]]
    benign_scores = [r["score"] for r in validations if not r["worker_malicious"]]
    rejected_malicious = [
        r for r in validations
        if r["worker_malicious"] and r["rejected_by_threshold"]
    ]
    accepted_malicious = [
        r for r in validations
        if r["worker_malicious"] and r["accepted_for_aggregation"]
    ]
    rejected_benign = [
        r for r in validations
        if not r["worker_malicious"] and r["rejected_by_threshold"]
    ]
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").exists() else []
    return {
        "id": run_spec["id"],
        "tag": run_spec["tag"],
        "run": run_dir.name,
        "wall_seconds": wall_seconds,
        "rounds": len(timings),
        "validation_records": len(validations),
        "malicious_records": len(malicious_scores),
        "benign_records": len(benign_scores),
        "malicious_score_mean": (sum(malicious_scores) / len(malicious_scores)) if malicious_scores else None,
        "malicious_score_min": min(malicious_scores) if malicious_scores else None,
        "malicious_score_max": max(malicious_scores) if malicious_scores else None,
        "benign_score_mean": (sum(benign_scores) / len(benign_scores)) if benign_scores else None,
        "benign_score_max": max(benign_scores) if benign_scores else None,
        "malicious_rejected": len(rejected_malicious),
        "malicious_accepted": len(accepted_malicious),
        "benign_rejected": len(rejected_benign),
        "decisions": len(decisions),
        "checkpoints": [str(path.relative_to(PROJECT_ROOT)) for path in checkpoints],
    }


def run_one(run_spec: dict, common: list[str], results_path: Path, timeout_seconds: int) -> dict | None:
    before = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    run_tag = run_spec["id"].replace("/", "_").replace(" ", "_")
    existing = [name for name in before if name.endswith("_" + run_tag)]
    if existing:
        raise RuntimeError(f"run tag already exists; refusing duplicate execution: {run_tag}")
    command = [
        str(VENV_PYTHON), str(BFL_DIR / "main.py"),
        *common, "--run_tag", run_tag, *run_spec["args"],
    ]
    start = time.time()
    try:
        process = subprocess.run(
            command, cwd=BFL_DIR, capture_output=True, text=True, timeout=timeout_seconds
        )
        returncode = process.returncode
        stderr_tail = process.stderr[-800:]
    except subprocess.TimeoutExpired:
        returncode = "timeout"
        stderr_tail = "timeout"
    wall_seconds = time.time() - start

    # Match the run's own directory by its unique tag suffix: concurrent runs
    # may finish in any order, so the newest directory is not reliable.
    created = sorted(
        name for name in ({path.name for path in LOGS_DIR.iterdir() if path.is_dir()} - before)
        if name.endswith("_" + run_tag)
    )
    if returncode != 0 or not created:
        print(f"{run_spec['id']}: FAILED rc={returncode} created={created} {stderr_tail}")
        return None

    run_dir = LOGS_DIR / created[0]
    summary = summarize_run(run_dir, run_spec, wall_seconds)
    # Keep the absolute log directory so results can be merged across machines.
    summary["log_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
    with RESULTS_LOCK:
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(f"{run_spec['id']}: done in {wall_seconds:.0f}s -> {run_dir.name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--run-ids", type=str, default=None, help="comma separated run ids to execute")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runs = manifest["runs"]
    if args.run_ids:
        wanted = {item.strip() for item in args.run_ids.split(",") if item.strip()}
        runs = [run for run in runs if run["id"] in wanted]
    else:
        runs = [run for index, run in enumerate(runs) if index % args.shards == args.shard]

    args.results.parent.mkdir(parents=True, exist_ok=True)
    if not runs:
        print("no runs assigned")
        return

    print(f"executing {len(runs)} runs with {args.workers} workers (shard {args.shard}/{args.shards})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_one, run, manifest["common"], args.results, args.timeout_seconds)
            for run in runs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"results appended to {args.results}")


if __name__ == "__main__":
    main()

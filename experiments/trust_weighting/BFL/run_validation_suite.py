"""Run the 3-node pipeline-validation suite (not formal paper results).

Steps
1. Calibration: clean runs collect benign validator scores and a p99 threshold.
2. Suite: attack and clean runs across the four blockchain/defense modes.
3. Summary: detection/aggregation statistics and timing records per run.

Every run is a short 3-node simulation that exercises the full code path; the
outputs prove that logs, checkpoints and ablations work end to end.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import statistics
import subprocess
from pathlib import Path

BFL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BFL_DIR.parent
VENV_PYTHON = Path(os.environ.get("BFL_PYTHON", sys.executable))
LOGS_DIR = PROJECT_ROOT / "logs" / "blogs"


def run_simulation(tag: str, extra_args: list[str], timeout_seconds: int) -> Path | None:
    command = [
        str(VENV_PYTHON),
        str(BFL_DIR / "main.py"),
        "-nd", "3",
        "-ha", "1,1,1",
        "-aio", "1",
        "-pow", "0",
        "-ko", "2",
        "-cs", "0",
        "-B", "128",
        "-mn", "ddpm",
        "-iid", "0",
        "-lr", "0.001",
        "-dtx", "1",
        "-le", "1",
        "-v", "0",
        "--test_batchsize", "256",
        "--save_global_checkpoints", "1",
        "--checkpoint_frequency", "5",
        *extra_args,
    ]
    before = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    process = subprocess.run(
        command, cwd=BFL_DIR, capture_output=True, text=True, timeout=timeout_seconds
    )
    created = sorted({path.name for path in LOGS_DIR.iterdir() if path.is_dir()} - before)
    if process.returncode != 0 or not created:
        print(f"{tag}: FAILED rc={process.returncode} created={created}")
        print(process.stderr[-1500:])
        return None
    print(f"{tag}: {created[-1]}")
    return LOGS_DIR / created[-1]


def run_suite(args, threshold: float) -> list[dict]:
    suite_runs = []
    for index in range(args.suite_seeds):
        seed = 9700 + index
        variant = [
            ("clean_bc1_def1", ["-nm", "0", "--blockchain_on", "1", "--defense_on", "1"]),
            ("attack_bc1_def1", ["-nm", "1", "-nv", str(args.noise_scale), "--blockchain_on", "1", "--defense_on", "1"]),
            ("attack_bc0_def1", ["-nm", "1", "-nv", str(args.noise_scale), "--blockchain_on", "0", "--defense_on", "1"]),
            ("attack_bc0_def0", ["-nm", "1", "-nv", str(args.noise_scale), "--blockchain_on", "0", "--defense_on", "0"]),
        ]
        for tag, mode_args in variant:
            full_tag = f"{tag}-{seed}"
            run_dir = run_simulation(
                full_tag,
                [
                    "-max_ncomm", str(args.rounds),
                    "-vh", str(threshold),
                    "--seed", str(seed),
                    "--validation_seed", "9001",
                    *mode_args,
                ],
                args.timeout_seconds,
            )
            if run_dir is not None:
                suite_runs.append(summarize_run(run_dir, full_tag))
    return suite_runs


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_run(run_dir: Path, tag: str) -> dict:
    validations = read_jsonl(run_dir / "validation_records.jsonl")
    decisions = read_jsonl(run_dir / "aggregation_decisions.jsonl")
    timings = read_jsonl(run_dir / "timing_records.jsonl")
    malicious_scores = [r["score"] for r in validations if r["worker_malicious"]]
    benign_scores = [r["score"] for r in validations if not r["worker_malicious"]]
    malicious_decisions = [r for r in decisions if r["worker_malicious"]]
    benign_decisions = [r for r in decisions if not r["worker_malicious"]]

    def unique_accepted(rows):
        accepted_by_pair = {}
        for row in rows:
            key = (row["observer_id"], row["worker_id"], row["round"])
            accepted_by_pair.setdefault(key, False)
            accepted_by_pair[key] = accepted_by_pair[key] or bool(row["accepted_for_aggregation"])
        return accepted_by_pair

    malicious_pairs = {
        (row["worker_id"], row["round"]): True for row in malicious_decisions
    }
    malicious_accepted = {
        (row["worker_id"], row["round"]): False for row in malicious_decisions
    }
    for key, accepted in unique_accepted(malicious_decisions).items():
        pair = (key[1], key[2])
        malicious_accepted[pair] = malicious_accepted.get(pair, False) or accepted
    benign_pairs = {
        (row["worker_id"], row["round"]): True for row in benign_decisions
    }
    benign_rejected = {
        (row["worker_id"], row["round"]): False for row in benign_decisions
    }
    for key, accepted in unique_accepted(benign_decisions).items():
        pair = (key[1], key[2])
        benign_rejected[pair] = benign_rejected.get(pair, False) or (not accepted)
    return {
        "tag": tag,
        "run": run_dir.name,
        "rounds": len(timings),
        "malicious_validations": len(malicious_scores),
        "benign_validations": len(benign_scores),
        "malicious_scores": malicious_scores,
        "benign_scores": benign_scores,
        "malicious_updates_accepted": sum(malicious_accepted.values()),
        "malicious_updates_total": len(malicious_pairs),
        "benign_updates_rejected": sum(benign_rejected.values()),
        "benign_updates_total": len(benign_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--calibration-runs", type=int, default=6)
    parser.add_argument("--suite-seeds", type=int, default=3)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--threshold-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--suite-only", action="store_true")
    args = parser.parse_args()

    calibration_scores: list[float] = []

    if args.suite_only:
        threshold_document = json.loads(args.threshold_output.read_text(encoding="utf-8"))
        threshold = threshold_document["p99"]
        print(f"loaded threshold from {args.threshold_output}: {threshold}")
        suite_runs = run_suite(args, threshold)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps({
            "threshold": threshold_document,
            "noise_scale": args.noise_scale,
            "rounds": args.rounds,
            "runs": suite_runs,
        }, indent=2) + "\n")
        print(f"wrote summary for {len(suite_runs)} runs to {args.summary_output}")
        return

    # Step 1: calibration on clean runs only.
    for index in range(args.calibration_runs):
        seed = 9500 + index
        run_dir = run_simulation(
            f"calibrate-{seed}",
            [
                "-max_ncomm", str(args.rounds),
                "-nm", "0",
                "-vh", "999",
                "--seed", str(seed),
                "--validation_seed", "9001",
                "--blockchain_on", "1",
                "--defense_on", "1",
            ],
            args.timeout_seconds,
        )
        if run_dir is None:
            continue
        for record in read_jsonl(run_dir / "validation_records.jsonl"):
            if not record["worker_malicious"]:
                calibration_scores.append(record["score"])

    if not calibration_scores:
        raise SystemExit("no calibration scores collected")
    sorted_scores = sorted(calibration_scores)
    p99_index = min(len(sorted_scores) - 1, int(0.99 * len(sorted_scores)))
    threshold = sorted_scores[p99_index]
    threshold_document = {
        "rule": "benign p99 from independent clean runs",
        "samples": len(sorted_scores),
        "mean": statistics.mean(sorted_scores),
        "stdev": statistics.pstdev(sorted_scores),
        "min": sorted_scores[0],
        "max": sorted_scores[-1],
        "p99": threshold,
        "runs": args.calibration_runs,
        "rounds_per_run": args.rounds,
    }
    args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_output.write_text(json.dumps(threshold_document, indent=2) + "\n")
    print(f"calibrated threshold: {threshold_document}")

    if args.calibration_only:
        return

    # Step 2: mode matrix with the frozen threshold.
    suite_runs = run_suite(args, threshold)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps({
        "threshold": threshold_document,
        "noise_scale": args.noise_scale,
        "rounds": args.rounds,
        "runs": suite_runs,
    }, indent=2) + "\n")
    print(f"wrote summary for {len(suite_runs)} runs to {args.summary_output}")


if __name__ == "__main__":
    main()

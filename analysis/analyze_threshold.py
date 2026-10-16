"""Freeze the validator threshold from collected clean calibration runs."""

from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, required=True, help="runs.jsonl from collect_runs.py"
    )
    parser.add_argument("--tag", default="calibration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()
    benign_scores = []
    run_ids = []
    for record in read_jsonl(args.runs):
        if record["tag"] != args.tag:
            continue
        run_ids.append(record["run_id"])
        run_dir = project_root / record["log_dir"]
        for validation in read_jsonl(run_dir / "validation_records.jsonl"):
            if not validation["worker_malicious"] and (
                not validation["validator_malicious"]
            ):
                benign_scores.append(validation["score"])
    if not benign_scores:
        raise SystemExit("no benign calibration scores found")
    scores = sorted(benign_scores)
    index = min(len(scores) - 1, int(args.percentile / 100.0 * len(scores)))
    document = {
        "rule": f"benign p{args.percentile:g} from clean calibration runs",
        "runs": run_ids,
        "samples": len(scores),
        "mean": statistics.mean(scores),
        "stdev": statistics.pstdev(scores),
        "min": scores[0],
        "max": scores[-1],
        "p90": scores[min(len(scores) - 1, int(0.9 * len(scores)))],
        "p95": scores[min(len(scores) - 1, int(0.95 * len(scores)))],
        "p99": scores[index],
        "threshold": scores[index],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()

"""Choose the largest attack scale whose mean worker-round TPR is below 0.9."""

from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tpr", type=float, default=0.9)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for record in read_jsonl(args.runs):
        if not record["tag"].startswith("sweep_sigma_"):
            continue
        total = record.get("malicious_worker_round_total", 0)
        if total <= 0:
            continue
        scale = float(record["tag"].removeprefix("sweep_sigma_"))
        tpr = record.get("malicious_worker_round_filter_rejected", 0) / total
        grouped[scale].append(tpr)
    table = []
    for scale, rates in sorted(grouped.items()):
        table.append(
            {
                "sigma": scale,
                "runs": len(rates),
                "tpr_mean": statistics.mean(rates),
                "tpr_std": statistics.pstdev(rates) if len(rates) > 1 else None,
                "tpr_min": min(rates),
                "tpr_max": max(rates),
            }
        )
    eligible = [row for row in table if row["tpr_mean"] < args.max_tpr]
    if eligible:
        selected = max(eligible, key=lambda row: row["sigma"])
    elif table:
        selected = min(table, key=lambda row: row["sigma"])
    else:
        raise SystemExit("no malicious worker-round records found")
    result = {
        "rule": f"largest sigma with mean worker-round TPR < {args.max_tpr}",
        "selected": selected,
        "table": table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

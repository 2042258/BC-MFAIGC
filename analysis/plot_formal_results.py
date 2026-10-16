#!/usr/bin/env python3
"""Create compact statistical tables and figures for the released formal runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values) if len(
        values
    ) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--fid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = read_jsonl(args.runs)
    fids = read_jsonl(args.fid) if args.fid.exists() else []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        grouped[row["tag"]].append(row)

    rows = []
    for tag, items in sorted(grouped.items()):
        det = [
            r["malicious_worker_round_filter_rejected"]
            / r["malicious_worker_round_total"]
            for r in items
            if r["malicious_worker_round_total"]
        ]
        fpr = [
            r["benign_worker_round_filter_rejected"] / r["benign_worker_round_total"]
            for r in items
            if r["benign_worker_round_total"]
        ]
        score_m = [
            r["malicious_score_mean"]
            for r in items
            if r["malicious_score_mean"] is not None
        ]
        score_b = [
            r["benign_score_mean"] for r in items if r["benign_score_mean"] is not None
        ]
        # The centralized DDPM reference record has no federated round-time
        # field; keep it in the grouped summary with a missing timing value.
        round_time = [
            r.get("round_wall_mean_seconds")
            for r in items
            if r.get("round_wall_mean_seconds") is not None
        ]
        det_m, det_s = mean_std(det)
        fpr_m, fpr_s = mean_std(fpr)
        rows.append(
            {
                "tag": tag,
                "runs": len(items),
                "malicious_detection_mean": det_m,
                "malicious_detection_std": det_s,
                "benign_false_positive_mean": fpr_m,
                "benign_false_positive_std": fpr_s,
                "malicious_score_mean": statistics.mean(score_m) if score_m else None,
                "benign_score_mean": statistics.mean(score_b) if score_b else None,
                "round_wall_mean_seconds": statistics.mean(round_time)
                if round_time
                else None,
            }
        )

    csv_path = args.output_dir / "formal_statistics_20260927.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["tag"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "formal_statistics_20260927.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    labels = [r["tag"] for r in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5.5))
    ax.bar(
        [i - 0.2 for i in x],
        [r["malicious_detection_mean"] or 0 for r in rows],
        width=0.4,
        label="Malicious detection",
    )
    ax.bar(
        [i + 0.2 for i in x],
        [r["benign_false_positive_mean"] or 0 for r in rows],
        width=0.4,
        label="Benign false positive",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "formal_detection_rates_20260927.png", dpi=220)
    plt.close(fig)

    if fids:
        fid_groups: dict[str, list[float]] = defaultdict(list)
        for row in fids:
            fid_groups[row["tag"]].append(float(row["fid"]))
        fid_labels = sorted(fid_groups)
        fid_values = [statistics.mean(fid_groups[label]) for label in fid_labels]
        fig, ax = plt.subplots(figsize=(max(10, len(fid_labels) * 0.7), 5.5))
        ax.bar(range(len(fid_labels)), fid_values)
        ax.set_ylabel("FID")
        ax.set_xticks(range(len(fid_labels)), fid_labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "formal_fid_comparison_20260927.png", dpi=220)
        plt.close(fig)

    print(
        f"wrote {len(rows)} grouped rows and {len(fids)} FID rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

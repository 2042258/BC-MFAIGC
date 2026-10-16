"""Build final detection/FID/timing tables and per-round figure data."""

from __future__ import annotations
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

FEDERATED_RUN_FILES = {
    "core": "phase1_validation/runs_core_100r_v4.jsonl",
    "uptime2": "phase1_validation/runs_uptime2_only.jsonl",
    "noniid": "phase1_validation/runs_noniid_validation_v4.jsonl",
    "strong": "phase1_validation/runs_strong_only.jsonl",
}
TIMING_RUN_FILE = "phase1_validation/runs_timing_v4.jsonl"
DEVICE_LINE = re.compile(
    "^device_\\d+ (worker|miner|validator) [BM]:\\s*([0-9.eE+-]+)\\s*$"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean_std(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def accuracy_series(run_dir: Path, max_rounds: int = 100) -> dict[int, float]:
    """Per-round global-model loss.

    The median across devices is used instead of the mean so that a device
    that was blacklisted mid-run and keeps a stale model does not skew the
    round value. For the main matrix all devices are identical, so the choice
    does not change the published numbers.
    """
    series = {}
    for round_no in range(1, max_rounds + 1):
        path = run_dir / f"comm_{round_no}" / f"accuracy_comm_{round_no}.txt"
        if not path.exists():
            continue
        values = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = DEVICE_LINE.match(line.strip())
            if match:
                values.append(float(match.group(2)))
        if values:
            series[round_no] = statistics.median(values)
    return series


def detection_table(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["tag"]].append(record)
    table = {}
    for tag, rows in sorted(groups.items()):
        tp = sum((r.get("malicious_worker_round_filter_rejected", 0) for r in rows))
        malicious_total = sum((r.get("malicious_worker_round_total", 0) for r in rows))
        fp = sum((r.get("benign_worker_round_filter_rejected", 0) for r in rows))
        benign_total = sum((r.get("benign_worker_round_total", 0) for r in rows))
        applied_malicious = sum(
            (r.get("malicious_worker_round_applied", 0) for r in rows)
        )
        not_agg_malicious = sum(
            (r.get("malicious_worker_round_not_aggregated", 0) for r in rows)
        )
        run_tpr = [
            r["malicious_worker_round_filter_rejected"]
            / r["malicious_worker_round_total"]
            for r in rows
            if r.get("malicious_worker_round_total")
        ]
        run_fpr = [
            r["benign_worker_round_filter_rejected"] / r["benign_worker_round_total"]
            for r in rows
            if r.get("benign_worker_round_total")
        ]
        table[tag] = {
            "runs": len(rows),
            "malicious_rounds": malicious_total,
            "malicious_filter_rejected": tp,
            "malicious_filter_accepted": malicious_total - tp,
            "malicious_applied": applied_malicious,
            "malicious_not_aggregated": not_agg_malicious,
            "benign_rounds": benign_total,
            "benign_filter_rejected": fp,
            "tpr_detection_rate": mean_std(run_tpr),
            "fnr_miss_rate": mean_std([1.0 - v for v in run_tpr])
            if run_tpr
            else {"n": 0, "mean": None, "std": None},
            "fpr_false_positive_rate": mean_std(run_fpr),
            "pooled_tpr": tp / malicious_total if malicious_total else None,
            "pooled_fpr": fp / benign_total if benign_total else None,
            "pooled_malicious_applied_rate": applied_malicious / malicious_total
            if malicious_total
            else None,
        }
    return table


def central_tag(run_id: str) -> str:
    parts = run_id.split("-")
    if run_id.startswith("tra-clean"):
        return "tra_clean"
    if run_id.startswith("tra-contam") and len(parts) >= 3:
        return f"tra_contam_{parts[2]}"
    return run_id


def fid_table(fid_records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in fid_records:
        tag = record["tag"]
        run_id = record.get("id") or record.get("run_id") or ""
        if run_id.startswith("tra-"):
            tag = central_tag(run_id)
        groups[tag].append(record["fid"])
    return {tag: mean_std(values) for tag, values in sorted(groups.items())}


def timing_table(federated_records: list[dict], *, project_root=None) -> dict:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    groups = defaultdict(lambda: defaultdict(list))
    for record in federated_records:
        run_dir = project_root / record["log_dir"]
        tag = record["tag"]
        for timing in read_jsonl(run_dir / "timing_records.jsonl"):
            if timing.get("phase") == "round_summary":
                if timing.get("wall_clock_seconds") is not None:
                    groups[tag]["round_wall_seconds"].append(
                        float(timing["wall_clock_seconds"])
                    )
                if timing.get("block_generation_seconds") is not None:
                    groups[tag]["block_generation_seconds"].append(
                        float(timing["block_generation_seconds"])
                    )
                continue
            for value in timing.get("worker_local_training_times", {}).values():
                if isinstance(value, (int, float)):
                    groups[tag]["worker_local_training_seconds"].append(float(value))
            for value in timing.get("validator_reference_update_times", {}).values():
                if isinstance(value, (int, float)):
                    groups[tag]["validator_reference_seconds"].append(float(value))
            for transaction in timing.get("transactions", []):
                value = transaction.get("modelled_payload_transmission_seconds")
                if isinstance(value, (int, float)):
                    groups[tag]["modelled_transmission_seconds"].append(float(value))
                value = transaction.get("validation_time")
                if isinstance(value, (int, float)):
                    groups[tag]["validation_seconds"].append(float(value))
    return {
        tag: {name: mean_std(values) for name, values in sorted(metrics.items())}
        for tag, metrics in sorted(groups.items())
    }


def figure_data(
    federated_records: list[dict],
    central_records: list[dict],
    central_epochs: int,
    *,
    project_root=None,
) -> dict:
    """Per-round mean/std series for the new Figure 10-14 set."""
    project_root = Path.cwd() if project_root is None else Path(project_root)
    federated = defaultdict(lambda: defaultdict(list))
    for record in federated_records:
        run_dir = project_root / record["log_dir"]
        series = accuracy_series(run_dir)
        for round_no, value in series.items():
            federated[record["tag"]][round_no].append(value)
    federated_series = {
        tag: {
            str(round_no): mean_std(values)
            for round_no, values in sorted(rounds.items())
        }
        for tag, rounds in sorted(federated.items())
    }
    central = defaultdict(lambda: defaultdict(list))
    for record in central_records:
        run_dir = project_root / record["log_dir"]
        for row in read_jsonl(run_dir / "central_training_records.jsonl"):
            central[record["tag"]][int(row["epoch"])].append(
                float(row["validation_loss"])
            )
    central_series = {
        tag: {str(epoch): mean_std(values) for epoch, values in sorted(rounds.items())}
        for tag, rounds in sorted(central.items())
    }
    final_federated = {}
    for tag, rounds in federated_series.items():
        last_round = max((int(key) for key in rounds))
        final_federated[tag] = rounds[str(last_round)]
    final_central = {}
    for tag, epochs in central_series.items():
        last_epoch = max((int(key) for key in epochs))
        final_central[tag] = epochs[str(last_epoch)]
    return {
        "per_round_federated": federated_series,
        "per_epoch_central": central_series,
        "final_federated": final_federated,
        "final_central": final_central,
        "central_epochs": central_epochs,
    }


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "phase1_validation" / "reports",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    federated_records = []
    for name, relative in FEDERATED_RUN_FILES.items():
        federated_records.extend(read_jsonl(project_root / relative))
    timing_records = read_jsonl(project_root / TIMING_RUN_FILE)
    central_records = read_jsonl(project_root / "phase1_validation/runs_tra_v4.jsonl")
    fid_records = read_jsonl(project_root / "phase1_validation/fid_final_v4.jsonl")
    calibration_records = read_jsonl(
        project_root / "phase1_validation/runs_calib100_v4.jsonl"
    )
    threshold_document = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/comparison/phase1_validation/frozen_threshold_100r_v4.json"
        ).read_text()
    )
    sigma_document = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/comparison/phase1_validation/frozen_sigma_100r_v4.json"
        ).read_text()
    )
    central_epochs = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/comparison/phase1_validation/matrix_tra_v4.json"
        ).read_text()
    )["runs"][0]["args"]
    central_epochs = int(central_epochs[central_epochs.index("--epochs") + 1])
    reports = {
        "threshold": threshold_document,
        "sigma_selection": sigma_document,
        "central_epochs": central_epochs,
        "detection": detection_table(federated_records),
        "fid": fid_table(fid_records),
        "timing_single_tenant": timing_table(timing_records, project_root=project_root),
        "figures": figure_data(
            federated_records,
            central_records,
            central_epochs,
            project_root=project_root,
        ),
    }
    (args.output_dir / "final_reports.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote reports to {args.output_dir / 'final_reports.json'}")
    print(
        f"federated runs: {len(federated_records)}, timing runs: {len(timing_records)}, central runs: {len(central_records)}, fid: {len(fid_records)}"
    )


if __name__ == "__main__":
    main()

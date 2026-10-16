"""Aggregate collected runs into detection, aggregation and timing tables."""

from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_timings(run_dir: Path) -> dict:
    timings = read_jsonl(run_dir / "timing_records.jsonl")
    local_times = []
    training_times = []
    validation_times = []
    transmission_times = []
    reference_times = []
    round_times = []
    for record in timings:
        for value in record.get("worker_local_update_times", {}).values():
            if isinstance(value, (int, float)):
                local_times.append(float(value))
        for value in record.get("worker_local_training_times", {}).values():
            if isinstance(value, (int, float)):
                training_times.append(float(value))
        for value in record.get("validator_reference_update_times", {}).values():
            if isinstance(value, (int, float)):
                reference_times.append(float(value))
        for transaction in record.get("transactions", []):
            value = transaction.get("validation_time")
            if isinstance(value, (int, float)):
                validation_times.append(float(value))
            value = transaction.get("modelled_payload_transmission_seconds")
            if isinstance(value, (int, float)):
                transmission_times.append(float(value))
        if record.get("phase") == "round_summary" and isinstance(
            record.get("wall_clock_seconds"), (int, float)
        ):
            round_times.append(float(record["wall_clock_seconds"]))
    return {
        "rounds": len(round_times),
        "local_update_mean": statistics.mean(local_times) if local_times else None,
        "local_training_mean": statistics.mean(training_times)
        if training_times
        else None,
        "reference_update_mean": statistics.mean(reference_times)
        if reference_times
        else None,
        "validation_mean": statistics.mean(validation_times)
        if validation_times
        else None,
        "transmission_model_mean": statistics.mean(transmission_times)
        if transmission_times
        else None,
        "round_wall_mean": statistics.mean(round_times) if round_times else None,
    }


def mean_or_none(values):
    return statistics.mean(values) if values else None


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "runs": 0,
            "rounds": 0,
            "malicious_vote_total": 0,
            "malicious_vote_rejected": 0,
            "benign_vote_total": 0,
            "benign_vote_rejected": 0,
            "malicious_worker_round_total": 0,
            "malicious_worker_round_filter_rejected": 0,
            "malicious_worker_round_filter_accepted": 0,
            "malicious_worker_round_applied": 0,
            "benign_worker_round_total": 0,
            "benign_worker_round_filter_rejected": 0,
            "benign_worker_round_applied": 0,
            "malicious_worker_round_not_aggregated": 0,
            "benign_worker_round_not_aggregated": 0,
            "malicious_score_means": [],
            "benign_score_means": [],
            "run_malicious_vote_rates": [],
            "run_benign_vote_fpr": [],
            "run_malicious_final_rates": [],
            "run_benign_final_fpr": [],
            "local_update_means": [],
            "local_training_means": [],
            "reference_update_means": [],
            "validation_means": [],
            "transmission_model_means": [],
            "round_wall_means": [],
            "observer_disagreements": 0,
        }
    )
    records = []
    for runs_path in args.runs:
        records.extend(read_jsonl(runs_path))
    for record in records:
        group = groups[record["tag"]]
        group["runs"] += 1
        group["rounds"] = max(group["rounds"], record["rounds"])
        for key in (
            "malicious_vote_total",
            "malicious_vote_rejected",
            "benign_vote_total",
            "benign_vote_rejected",
            "malicious_worker_round_total",
            "malicious_worker_round_filter_rejected",
            "malicious_worker_round_filter_accepted",
            "malicious_worker_round_applied",
            "benign_worker_round_total",
            "benign_worker_round_filter_rejected",
            "benign_worker_round_applied",
            "malicious_worker_round_not_aggregated",
            "benign_worker_round_not_aggregated",
            "observer_disagreements",
        ):
            group[key] += record.get(key, 0)
        if record["malicious_score_mean"] is not None:
            group["malicious_score_means"].append(record["malicious_score_mean"])
        if record["benign_score_mean"] is not None:
            group["benign_score_means"].append(record["benign_score_mean"])
        if record["malicious_vote_total"]:
            group["run_malicious_vote_rates"].append(
                record["malicious_vote_rejected"] / record["malicious_vote_total"]
            )
        if record["benign_vote_total"]:
            group["run_benign_vote_fpr"].append(
                record["benign_vote_rejected"] / record["benign_vote_total"]
            )
        if record["malicious_worker_round_total"]:
            group["run_malicious_final_rates"].append(
                record["malicious_worker_round_filter_rejected"]
                / record["malicious_worker_round_total"]
            )
        if record["benign_worker_round_total"]:
            group["run_benign_final_fpr"].append(
                record["benign_worker_round_filter_rejected"]
                / record["benign_worker_round_total"]
            )
        timings = collect_timings(project_root / record["log_dir"])
        if timings["local_update_mean"] is not None:
            group["local_update_means"].append(timings["local_update_mean"])
        if timings["local_training_mean"] is not None:
            group["local_training_means"].append(timings["local_training_mean"])
        if timings["reference_update_mean"] is not None:
            group["reference_update_means"].append(timings["reference_update_mean"])
        if timings["validation_mean"] is not None:
            group["validation_means"].append(timings["validation_mean"])
        if timings["transmission_model_mean"] is not None:
            group["transmission_model_means"].append(timings["transmission_model_mean"])
        if timings["round_wall_mean"] is not None:
            group["round_wall_means"].append(timings["round_wall_mean"])
    summary = {}
    for tag, group in sorted(groups.items()):
        summary[tag] = {
            "runs": group["runs"],
            "rounds": group["rounds"],
            "malicious_vote_detection_rate": group["malicious_vote_rejected"]
            / group["malicious_vote_total"]
            if group["malicious_vote_total"]
            else None,
            "malicious_worker_round_detection_rate": group[
                "malicious_worker_round_filter_rejected"
            ]
            / group["malicious_worker_round_total"]
            if group["malicious_worker_round_total"]
            else None,
            "malicious_worker_round_accept_rate": group[
                "malicious_worker_round_filter_accepted"
            ]
            / group["malicious_worker_round_total"]
            if group["malicious_worker_round_total"]
            else None,
            "benign_vote_false_positive_rate": group["benign_vote_rejected"]
            / group["benign_vote_total"]
            if group["benign_vote_total"]
            else None,
            "benign_worker_round_false_positive_rate": group[
                "benign_worker_round_filter_rejected"
            ]
            / group["benign_worker_round_total"]
            if group["benign_worker_round_total"]
            else None,
            "malicious_worker_round_filter_rejected": group[
                "malicious_worker_round_filter_rejected"
            ],
            "malicious_worker_round_filter_accepted": group[
                "malicious_worker_round_filter_accepted"
            ],
            "malicious_worker_round_applied": group["malicious_worker_round_applied"],
            "benign_worker_round_filter_rejected": group[
                "benign_worker_round_filter_rejected"
            ],
            "benign_worker_round_applied": group["benign_worker_round_applied"],
            "malicious_worker_round_not_aggregated": group[
                "malicious_worker_round_not_aggregated"
            ],
            "benign_worker_round_not_aggregated": group[
                "benign_worker_round_not_aggregated"
            ],
            "malicious_vote_rejected": group["malicious_vote_rejected"],
            "malicious_vote_total": group["malicious_vote_total"],
            "benign_vote_rejected": group["benign_vote_rejected"],
            "benign_vote_total": group["benign_vote_total"],
            "malicious_score_mean": mean_or_none(group["malicious_score_means"]),
            "benign_score_mean": mean_or_none(group["benign_score_means"]),
            "independent_run_n": group["runs"],
            "malicious_vote_rate_mean": mean_or_none(group["run_malicious_vote_rates"]),
            "malicious_vote_rate_std": statistics.pstdev(
                group["run_malicious_vote_rates"]
            )
            if len(group["run_malicious_vote_rates"]) > 1
            else None,
            "benign_vote_fpr_mean": mean_or_none(group["run_benign_vote_fpr"]),
            "benign_vote_fpr_std": statistics.pstdev(group["run_benign_vote_fpr"])
            if len(group["run_benign_vote_fpr"]) > 1
            else None,
            "malicious_final_rate_mean": mean_or_none(
                group["run_malicious_final_rates"]
            ),
            "malicious_final_rate_std": statistics.pstdev(
                group["run_malicious_final_rates"]
            )
            if len(group["run_malicious_final_rates"]) > 1
            else None,
            "benign_final_fpr_mean": mean_or_none(group["run_benign_final_fpr"]),
            "benign_final_fpr_std": statistics.pstdev(group["run_benign_final_fpr"])
            if len(group["run_benign_final_fpr"]) > 1
            else None,
            "local_update_mean_seconds": mean_or_none(group["local_update_means"]),
            "local_training_mean_seconds": mean_or_none(group["local_training_means"]),
            "reference_update_mean_seconds": mean_or_none(
                group["reference_update_means"]
            ),
            "validation_mean_seconds": mean_or_none(group["validation_means"]),
            "transmission_model_mean_seconds": mean_or_none(
                group["transmission_model_means"]
            ),
            "round_wall_mean_seconds": mean_or_none(group["round_wall_means"]),
            "observer_disagreements": group["observer_disagreements"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

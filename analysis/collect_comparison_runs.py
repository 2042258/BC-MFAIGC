"""Collect complete, protocol-tagged run summaries from experiment logs."""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TAG_PATTERNS = [
    (re.compile("^calib-(\\d+)$"), lambda m: "calibration"),
    (re.compile("^sweep-s([\\d.]+)-(\\d+)$"), lambda m: f"sweep_sigma_{m.group(1)}"),
    (
        re.compile("^matrix-nm(\\d+)-(bc\\d_def\\d)-(\\d+)$"),
        lambda m: f"nm{m.group(1)}_{m.group(2)}",
    ),
    (
        re.compile("^matrix-clean-(bc\\d_def\\d)-(\\d+)$"),
        lambda m: f"clean_{m.group(1)}",
    ),
    (
        re.compile("^uptime-nm(\\d+)-(bc\\d_def\\d)-(\\d+)$"),
        lambda m: f"uptime_nm{m.group(1)}_{m.group(2)}",
    ),
    (
        re.compile("^strong-nm(\\d+)-(bc\\d_def\\d)-(\\d+)$"),
        lambda m: f"strong_nm{m.group(1)}_{m.group(2)}",
    ),
    (
        re.compile("^uptime2-nm(\\d+)-(bc\\d_def\\d)-(\\d+)$"),
        lambda m: f"uptime2_nm{m.group(1)}_{m.group(2)}",
    ),
    (
        re.compile("^noniid_validation-nm(\\d+)-(bc\\d_def\\d)-(\\d+)$"),
        lambda m: f"noniid_nm{m.group(1)}_{m.group(2)}",
    ),
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_run_id(dir_name: str) -> tuple[str, str] | None:
    parts = dir_name.split("_", 2)
    if len(parts) != 3:
        return None
    run_id = parts[2]
    for pattern, tag_function in TAG_PATTERNS:
        match = pattern.match(run_id)
        if match:
            return (run_id, tag_function(match))
    return (run_id, "unknown")


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple((row[key] for key in keys))].append(row)
    return grouped


def _majority_bool(rows: list[dict], field: str) -> tuple[bool, bool]:
    values = [bool(row[field]) for row in rows]
    if not values:
        return (False, False)
    return (sum(values) > len(values) / 2, len(set(values)) > 1)


SHORT_OPTIONS = {
    "-nd": "num_devices",
    "-max_ncomm": "max_num_comm",
    "-nm": "num_malicious",
    "-nv": "noise_variance",
    "-vh": "validator_threshold",
    "-le": "default_local_epochs",
    "-B": "batchsize",
    "-iid": "IID",
    "-st": "shard_test_data",
    "-dts": "base_data_transmission_speed",
}
PROTOCOL_KEYS = {
    "seed",
    "num_devices",
    "max_num_comm",
    "num_malicious",
    "noise_variance",
    "validator_threshold",
    "default_local_epochs",
    "batchsize",
    "IID",
    "shard_test_data",
    "test_batchsize",
    "blockchain_on",
    "defense_on",
    "network_stability",
}


def _manifest_params(values: list[str]) -> dict[str, str]:
    params = {}
    index = 0
    while index < len(values):
        token = values[index]
        key = SHORT_OPTIONS.get(token, token[2:] if token.startswith("--") else None)
        if key and index + 1 < len(values) and (not values[index + 1].startswith("-")):
            params[key] = values[index + 1]
            index += 2
        else:
            index += 1
    return params


def _args_file_params(path: Path) -> dict[str, str]:
    params = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("--"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            params[parts[0][2:]] = parts[1].strip()
    return params


def _same_value(expected: str, actual: str | None) -> bool:
    if actual is None:
        return False
    if expected == actual:
        return True
    try:
        return float(expected) == float(actual)
    except ValueError:
        return False


def summarize(run_dir: Path, run_id: str, tag: str, *, project_root=None) -> dict:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    validations = read_jsonl(run_dir / "validation_records.jsonl")
    decisions = read_jsonl(run_dir / "aggregation_decisions.jsonl")
    applications = read_jsonl(run_dir / "aggregation_applications.jsonl")
    timings = read_jsonl(run_dir / "timing_records.jsonl")
    checkpoints = (
        sorted((run_dir / "checkpoints").glob("*.pt"))
        if (run_dir / "checkpoints").exists()
        else []
    )
    training_samples = 0
    for timing in timings:
        training_samples += sum(
            (
                value
                for value in timing.get("worker_training_samples", {}).values()
                if isinstance(value, (int, float))
            )
        )
    defense_on = None
    for timing in timings:
        if "defense_on" in timing:
            defense_on = bool(timing["defense_on"])
            break
    vote_rows = validations
    malicious_votes = [row for row in vote_rows if row["worker_malicious"]]
    benign_votes = [row for row in vote_rows if not row["worker_malicious"]]
    malicious_scores = [row["score"] for row in malicious_votes]
    benign_scores = [row["score"] for row in benign_votes]
    rejected_malicious_votes = sum(
        (
            not bool(row["accepted_for_aggregation"])
            if defense_on is not False
            else False
            for row in malicious_votes
        )
    )
    rejected_benign_votes = sum(
        (
            not bool(row["accepted_for_aggregation"])
            if defense_on is not False
            else False
            for row in benign_votes
        )
    )
    vote_groups = _group_rows(vote_rows, ("round", "worker_id", "worker_malicious"))
    decision_groups = _group_rows(decisions, ("round", "worker_id", "worker_malicious"))
    application_groups = _group_rows(
        applications, ("round", "worker_id", "worker_malicious")
    )
    worker_rounds = []
    observer_disagreements = 0
    all_worker_round_keys = set(vote_groups) | set(application_groups)
    for key in sorted(all_worker_round_keys, key=repr):
        round_no, worker_id, worker_malicious = key
        rows = vote_groups.get(key, [])
        decision_rows = decision_groups.get(key, [])
        application_rows = application_groups.get(key, [])
        if decision_rows:
            field = (
                "filter_accepted"
                if "filter_accepted" in decision_rows[0]
                else "accepted_for_aggregation"
            )
            filter_accepted, disagreement = _majority_bool(decision_rows, field)
            filter_rejected = not filter_accepted
            observer_disagreements += int(disagreement)
        elif rows:
            filter_rejected = None
            filter_accepted = None
        else:
            filter_rejected = None
            filter_accepted = None
        if application_rows:
            applied, disagreement = _majority_bool(
                application_rows, "aggregation_applied"
            )
            observer_disagreements += int(disagreement)
            status = "applied" if applied else "not_applied"
        else:
            applied = None
            status = "not_aggregated"
        worker_rounds.append(
            {
                "round": round_no,
                "worker_id": worker_id,
                "worker_malicious": worker_malicious,
                "filter_accepted": filter_accepted,
                "filter_rejected": filter_rejected,
                "aggregation_applied": applied,
                "status": status,
            }
        )
    malicious_worker_rounds = [row for row in worker_rounds if row["worker_malicious"]]
    benign_worker_rounds = [row for row in worker_rounds if not row["worker_malicious"]]
    malicious_filter_rows = [
        row for row in malicious_worker_rounds if row["filter_rejected"] is not None
    ]
    benign_filter_rows = [
        row for row in benign_worker_rounds if row["filter_rejected"] is not None
    ]
    malicious_filter_rejected = sum(
        (row["filter_rejected"] is True for row in malicious_filter_rows)
    )
    malicious_filter_accepted = sum(
        (row["filter_accepted"] is True for row in malicious_filter_rows)
    )
    malicious_applied = sum(
        (row["aggregation_applied"] is True for row in malicious_worker_rounds)
    )
    benign_filter_rejected = sum(
        (row["filter_rejected"] is True for row in benign_filter_rows)
    )
    benign_applied = sum(
        (row["aggregation_applied"] is True for row in benign_worker_rounds)
    )
    malicious_not_aggregated = sum(
        (row["status"] == "not_aggregated" for row in malicious_worker_rounds)
    )
    benign_not_aggregated = sum(
        (row["status"] == "not_aggregated" for row in benign_worker_rounds)
    )
    return {
        "run_id": run_id,
        "tag": tag,
        "run": run_dir.name,
        "log_dir": str(run_dir.resolve().relative_to(project_root.resolve())),
        "rounds": len(
            {row.get("round") for row in timings if row.get("phase") == "round_summary"}
        )
        or len(timings),
        "worker_training_samples": training_samples,
        "defense_on": defense_on,
        "validation_vote_records": len(vote_rows),
        "malicious_vote_total": len(malicious_votes),
        "malicious_vote_rejected": rejected_malicious_votes,
        "benign_vote_total": len(benign_votes),
        "benign_vote_rejected": rejected_benign_votes,
        "malicious_worker_round_total": len(malicious_filter_rows),
        "malicious_worker_round_filter_rejected": malicious_filter_rejected,
        "malicious_worker_round_filter_accepted": malicious_filter_accepted,
        "malicious_worker_round_applied": malicious_applied,
        "benign_worker_round_total": len(benign_filter_rows),
        "benign_worker_round_filter_rejected": benign_filter_rejected,
        "benign_worker_round_applied": benign_applied,
        "malicious_worker_round_not_aggregated": malicious_not_aggregated,
        "benign_worker_round_not_aggregated": benign_not_aggregated,
        "observer_disagreements": observer_disagreements,
        "malicious_score_mean": sum(malicious_scores) / len(malicious_scores)
        if malicious_scores
        else None,
        "malicious_score_max": max(malicious_scores) if malicious_scores else None,
        "benign_score_mean": sum(benign_scores) / len(benign_scores)
        if benign_scores
        else None,
        "benign_score_max": max(benign_scores) if benign_scores else None,
        "decisions": len(decisions),
        "checkpoints": [str(path.relative_to(project_root)) for path in checkpoints],
    }


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--logs-dir", type=Path, default=project_root / "logs" / "blogs"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional manifest; rejects unknown or incomplete run IDs",
    )
    args = parser.parse_args()
    args.logs_dir = args.logs_dir.resolve()
    expected_rounds = {}
    expected_params = {}
    manifest_tags = {}
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for spec in manifest.get("runs", []):
            manifest_tags[spec["id"]] = spec.get("tag", spec["id"])
            values = manifest.get("common", []) + spec.get("args", [])
            parsed = _manifest_params(values)
            expected_params[spec["id"]] = {
                key: parsed[key] for key in PROTOCOL_KEYS if key in parsed
            }
            expected = expected_params[spec["id"]].get("max_num_comm")
            expected_rounds[spec["id"]] = (
                int(expected) if expected is not None else None
            )
    records = []
    seen_ids = set()
    for run_dir in sorted((path for path in args.logs_dir.iterdir() if path.is_dir())):
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue
        run_id, tag = parsed
        if tag == "unknown":
            if run_id not in manifest_tags:
                continue
            tag = manifest_tags[run_id]
        if expected_rounds and run_id not in expected_rounds:
            continue
        record = summarize(run_dir, run_id, tag, project_root=project_root)
        expected = expected_rounds.get(run_id)
        if expected is not None and record["rounds"] != expected:
            continue
        if args.manifest is not None:
            actual = (
                _args_file_params(run_dir / "args_used.txt")
                if (run_dir / "args_used.txt").exists()
                else {}
            )
            mismatches = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in expected_params[run_id].items()
                if not _same_value(value, actual.get(key))
            }
            if mismatches:
                raise SystemExit(f"protocol mismatch for {run_id}: {mismatches}")
        if record["rounds"] == 0 and record["validation_vote_records"] == 0:
            continue
        if run_id in seen_ids:
            raise SystemExit(f"duplicate completed run ID found: {run_id}")
        seen_ids.add(run_id)
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join((json.dumps(record, sort_keys=True) for record in records)) + "\n",
        encoding="utf-8",
    )
    print(f"collected {len(records)} runs into {args.output}")


if __name__ == "__main__":
    main()

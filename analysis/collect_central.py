"""Collect centralized TRA run records into a stable JSONL index."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def central_tag(run_id: str) -> str:
    parts = run_id.split("-")
    if run_id.startswith("tra-clean"):
        return "tra_clean"
    if run_id.startswith("tra-contam") and len(parts) >= 3:
        return f"tra_contam_{parts[2]}"
    return run_id


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--logs-dir", type=Path, default=project_root / "logs" / "central"
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    args.logs_dir = args.logs_dir.resolve()
    allowed = None
    if args.manifest is not None:
        allowed = {
            spec["id"] for spec in json.loads(args.manifest.read_text()).get("runs", [])
        }
    collected = []
    for run_dir in sorted((path for path in args.logs_dir.iterdir() if path.is_dir())):
        run_id = run_dir.name.split("_", 2)[-1]
        if allowed is not None and run_id not in allowed:
            continue
        records_path = run_dir / "central_training_records.jsonl"
        if not records_path.exists():
            continue
        run_records = [
            json.loads(line)
            for line in records_path.read_text().splitlines()
            if line.strip()
        ]
        if not run_records:
            continue
        args_path = run_dir / "args_used.json"
        run_args = json.loads(args_path.read_text()) if args_path.exists() else {}
        checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
        collected.append(
            {
                "run_id": run_id,
                "id": run_id,
                "tag": central_tag(run_id),
                "run": run_dir.name,
                "log_dir": str(run_dir.resolve().relative_to(project_root.resolve())),
                "epochs": len(run_records),
                "final_validation_loss": run_records[-1]["validation_loss"],
                "noise_probability": run_args.get("noise_probability"),
                "noise_scale": run_args.get("noise_scale"),
                "seed": run_args.get("seed"),
                "checkpoints": [
                    str(path.relative_to(project_root)) for path in checkpoints
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join((json.dumps(record, sort_keys=True) for record in collected)) + "\n",
        encoding="utf-8",
    )
    print(f"collected {len(collected)} centralized runs into {args.output}")


if __name__ == "__main__":
    main()

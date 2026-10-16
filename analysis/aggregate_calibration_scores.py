"""Extract a compact, evidence-preserving snapshot from completed calibration logs."""

from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-glob", default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.logs_glob:
        paths = sorted(glob.glob(args.logs_glob))
    elif args.logs_dir:
        paths = sorted(args.logs_dir.glob("*inc-calib-*/validation_records.jsonl"))
    else:
        raise SystemExit("one of --logs-glob or --logs-dir is required")
    scores = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record["worker_malicious"] and (
                    not record["validator_malicious"]
                ):
                    scores.append(record["score"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores), encoding="utf-8")
    print(f"wrote {len(scores)} benign calibration scores")


if __name__ == "__main__":
    main()

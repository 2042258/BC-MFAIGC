#!/usr/bin/env python3
"""Collect the released core and ablation runs without relying on legacy run-id parsers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.collect_comparison_runs import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logs_dir = args.logs_dir.resolve()
    collected: list[dict] = []
    seen: set[str] = set()
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for spec in manifest.get("runs", []):
            run_id = spec["id"]
            if run_id in seen:
                raise SystemExit(f"duplicate run id: {run_id}")
            candidates = sorted(logs_dir.glob(f"*_{run_id}"))
            if len(candidates) != 1:
                raise SystemExit(
                    f"expected one completed log for {run_id}, found {len(candidates)}"
                )
            record = summarize(
                candidates[0],
                run_id,
                spec.get("tag", run_id),
                project_root=logs_dir.parents[1],
            )
            expected_rounds = (
                int(spec.get("args", [])[spec.get("args", []).index("-max_ncomm") + 1])
                if "-max_ncomm" in spec.get("args", [])
                else 100
            )
            if record["rounds"] != expected_rounds:
                raise SystemExit(
                    f"incomplete run {run_id}: {record['rounds']}/{expected_rounds} rounds"
                )
            record.update(
                {
                    "condition": spec.get("condition"),
                    "method": spec.get("method"),
                    "seed": spec.get("seed"),
                    "backend": spec.get("backend"),
                }
            )
            collected.append(record)
            seen.add(run_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in collected) + "\n",
        encoding="utf-8",
    )
    print(f"collected {len(collected)} released formal runs into {args.output}")


if __name__ == "__main__":
    main()

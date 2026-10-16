"""Run small calibration pilots and collect structured score records."""

from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

BFL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BFL_DIR.parent
VENV_PYTHON = Path(os.environ.get("BFL_PYTHON", sys.executable))
LOGS_DIR = PROJECT_ROOT / "logs" / "blogs"


def run_one(tag: str, extra_args: list[str], timeout_seconds: int) -> Path | None:
    command = [
        str(VENV_PYTHON),
        str(BFL_DIR / "main.py"),
        "-nd", "3",
        "-max_ncomm", "1",
        "-ha", "1,1,1",
        "-aio", "1",
        "-pow", "0",
        "-ko", "2",
        "-vh", "999",
        "-cs", "0",
        "-B", "128",
        "-mn", "ddpm",
        "-iid", "0",
        "-lr", "0.001",
        "-dtx", "1",
        "-le", "1",
        "-v", "0",
        "--save_global_checkpoints", "0",
        *extra_args,
    ]
    before = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    process = subprocess.run(
        command,
        cwd=BFL_DIR,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    after = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    created = sorted(after - before)
    if process.returncode != 0 or not created:
        print(f"{tag}: FAILED rc={process.returncode} created={created}")
        print(process.stderr[-2000:])
        return None
    print(f"{tag}: {created[-1]}")
    return LOGS_DIR / created[-1]


def collect(run_dir: Path, tag: str) -> list[dict]:
    records = []
    validation_path = run_dir / "validation_records.jsonl"
    if not validation_path.exists():
        return records
    for line in validation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["run"] = run_dir.name
        record["tag"] = tag
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-seeds", type=int, default=0)
    parser.add_argument("--attack-scale", type=float, default=None)
    parser.add_argument("--attack-seeds", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_records: list[dict] = []
    seed = args.seed_start

    if args.clean_seeds:
        for _ in range(args.clean_seeds):
            run_dir = run_one(
                f"clean-seed{seed}",
                ["-nm", "0", "--seed", str(seed), "--validation_seed", "9001"],
                args.timeout_seconds,
            )
            if run_dir is not None:
                output_records.extend(collect(run_dir, "clean"))
            seed += 1

    if args.attack_scale is not None and args.attack_seeds:
        for _ in range(args.attack_seeds):
            run_dir = run_one(
                f"attack-s{seed}",
                [
                    "-nm", "1",
                    "-nv", str(args.attack_scale),
                    "--seed", str(seed),
                    "--validation_seed", "9001",
                ],
                args.timeout_seconds,
            )
            if run_dir is not None:
                output_records.extend(collect(run_dir, f"attack_{args.attack_scale}"))
            seed += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in output_records) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()

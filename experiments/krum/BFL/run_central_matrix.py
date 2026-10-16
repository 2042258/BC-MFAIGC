"""Execute centralized TRA manifest runs with bounded parallelism."""

from __future__ import annotations

import os
import sys
import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
from pathlib import Path

BFL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BFL_DIR.parent
VENV_PYTHON = Path(os.environ.get("BFL_PYTHON", sys.executable))
LOGS_DIR = PROJECT_ROOT / "logs" / "central"
LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_one(spec: dict, results: Path, timeout: int) -> None:
    before = {p.name for p in LOGS_DIR.iterdir() if p.is_dir()} if LOGS_DIR.exists() else set()
    if any(name.endswith("_" + spec["id"]) for name in before):
        raise RuntimeError(f"central run tag already exists: {spec['id']}")
    command = [
        str(VENV_PYTHON), str(BFL_DIR / "train_central_ddpm.py"),
        *spec["args"], "--run-tag", spec["id"],
    ]
    started = time.time()
    process = subprocess.run(command, cwd=BFL_DIR, capture_output=True, text=True, timeout=timeout)
    after = {p.name for p in LOGS_DIR.iterdir() if p.is_dir()}
    suffix = "_" + spec["id"]
    created = [name for name in after - before if name.endswith(suffix)]
    if process.returncode != 0 or not created:
        raise RuntimeError(f"{spec['id']} failed rc={process.returncode}: {process.stderr[-1000:]}")
    run_dir = LOGS_DIR / created[0]
    records = read_jsonl(run_dir / "central_training_records.jsonl")
    result = {
        "id": spec["id"],
        "tag": spec["tag"],
        "run": run_dir.name,
        "log_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "epochs": len(records),
        "final_validation_loss": records[-1]["validation_loss"] if records else None,
        "wall_seconds": time.time() - started,
        "checkpoints": [str(p.relative_to(PROJECT_ROOT)) for p in sorted((run_dir / "checkpoints").glob("*.pt"))],
    }
    with LOCK:
        with results.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"{spec['id']}: done {result['wall_seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, spec, args.results, args.timeout_seconds) for spec in manifest["runs"]]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()

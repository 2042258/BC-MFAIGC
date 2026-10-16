"""Generate samples from final checkpoints and compute FID for each matrix run."""

from __future__ import annotations

import os
import sys
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BFL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BFL_DIR.parent
VENV_PYTHON = Path(os.environ.get("BFL_PYTHON", sys.executable))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--max-real-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-tags", type=str, default=None,
                        help="comma separated tag filter for the first pass")
    args = parser.parse_args()

    tag_filter = {tag.strip() for tag in args.only_tags.split(",")} if args.only_tags else None
    written = 0
    with args.output.open("a", encoding="utf-8") as output:
        for results_path in args.results:
            for record in read_jsonl(results_path):
                if tag_filter is not None and record["tag"] not in tag_filter:
                    continue
                if not record.get("checkpoints"):
                    continue
                checkpoint = PROJECT_ROOT / record["checkpoints"][-1]
                run_dir = PROJECT_ROOT / record["log_dir"]
                sample_dir = run_dir / "generated"
                fid_path = sample_dir / "fid.json"
                metadata_path = sample_dir / "evaluation_meta.json"

                run_id = record.get("id", record.get("run_id"))
                if run_id is None:
                    raise ValueError("run record lacks id/run_id")
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                sample_seed = 9000 + written
                expected_meta = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "num_samples": args.num_samples,
                    "max_real_samples": args.max_real_samples,
                    "batch_size": args.batch_size,
                    "sample_seed": sample_seed,
                }
                cached_meta = None
                if metadata_path.exists():
                    cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not fid_path.exists() or cached_meta != expected_meta:
                    subprocess.run(
                        [
                            str(VENV_PYTHON), str(BFL_DIR / "evaluate_ddpm.py"), str(checkpoint),
                            "--output-dir", str(sample_dir),
                            "--num-samples", str(args.num_samples),
                            "--batch-size", str(args.batch_size),
                            "--device", args.device,
                            "--seed", str(sample_seed),
                        ],
                        check=True,
                    )
                    subprocess.run(
                        [
                            str(VENV_PYTHON), str(BFL_DIR / "evaluate_fid.py"),
                            str(sample_dir / "samples.pt"),
                            "--output", str(fid_path),
                            "--max-real-samples", str(args.max_real_samples),
                            "--batch-size", str(args.batch_size),
                            "--device", args.device,
                        ],
                        check=True,
                    )
                    metadata_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")

                fid_document = json.loads(fid_path.read_text(encoding="utf-8"))
                output.write(json.dumps({
                    "id": run_id,
                    "tag": record["tag"],
                    "run": record["run"],
                    "checkpoint": record["checkpoints"][-1],
                    "num_samples": fid_document["generated_samples"],
                    "real_samples": fid_document["real_samples"],
                    "fid": fid_document["fid"],
                }, sort_keys=True) + "\n")
                output.flush()
                written += 1
                print(f"{run_id}: FID={fid_document['fid']:.3f}")
    print(f"wrote {written} FID records to {args.output}")


if __name__ == "__main__":
    main()

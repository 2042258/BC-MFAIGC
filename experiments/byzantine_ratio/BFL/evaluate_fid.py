"""Transitional FID entrypoint preserving this experiment's dataset root.

The implementation lives in bcmfaigc.evaluation.fid. Install the project first.
This entrypoint will be retired when the experiment runners are reorganized.
"""

from pathlib import Path

from bcmfaigc.evaluation.fid import main


if __name__ == "__main__":
    main(data_root=Path(__file__).resolve().parents[1] / "data")

"""Transitional centralized-training entrypoint retaining experiment dependencies.

Run from this BFL directory; dataset and log paths retain their original meaning.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from DatasetLoad import DatasetLoad
from models.Models import DDPM
from bcmfaigc.training.central import main


if __name__ == "__main__":
    main(dataset_type=DatasetLoad, model_type=DDPM)

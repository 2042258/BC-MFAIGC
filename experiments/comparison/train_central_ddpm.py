"""Historical root-path alias for the shared centralized DDPM trainer.

Run from BFL to retain the original relative dataset and log paths.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BFL"))

from DatasetLoad import DatasetLoad
from models.Models import DDPM
from bcmfaigc.training.central import main


if __name__ == "__main__":
    main(dataset_type=DatasetLoad, model_type=DDPM)

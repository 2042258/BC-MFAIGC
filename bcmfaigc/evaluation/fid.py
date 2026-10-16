"""Compute the standard FID for CIFAR-10 samples.

Uses the official FID Inception-v3 network from ``pytorch-fid`` (the same
feature extractor used by the reference implementation and most published
CIFAR-10 FID numbers). Generated samples are expected in ``[-1, 1]`` and are
converted to ``[0, 1]`` before feature extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

FEATURE_DIM = 2048


def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """FID with a scipy-compatible matrix square root (pytorch-fid 0.3.0
    still passes the removed ``disp`` argument to ``scipy.linalg.sqrtm``)."""
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
            raise ValueError(
                "FID covariance square root has a non-negligible imaginary part"
            )
        covmean = covmean.real
    mean_diff = mu1 - mu2
    return float(mean_diff @ mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def extract_inception_features(model, batches, device):
    features = []
    with torch.inference_mode():
        for images in batches:
            images = images.to(device, non_blocking=True).float().clamp(0, 1)
            features.append(model(images)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(features, axis=0)


def main(argv=None, *, data_root: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples", type=Path, help="samples.pt produced by evaluate_ddpm.py"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-real-samples", type=int, default=10000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_root if data_root is not None else Path("data"),
        help="Directory containing cifar-10-batches-py (no download is performed)",
    )
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.max_real_samples <= 1:
        raise ValueError(
            "batch-size must be positive and max-real-samples must exceed one"
        )

    device = torch.device(args.device)
    generated = torch.load(args.samples, map_location="cpu", weights_only=True)
    if isinstance(generated, dict):
        generated = generated["samples"]
    if generated.ndim != 4 or generated.shape[1:] != (3, 32, 32):
        raise ValueError("samples.pt must have shape [N, 3, 32, 32]")
    generated = ((generated.float() + 1.0) / 2.0).clamp(0, 1)
    generated_loader = DataLoader(
        generated, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    generated_batches = (images for images in generated_loader)

    real_dataset = CIFAR10(
        root=str(args.data_root),
        train=False,
        download=False,
        transform=lambda image: (
            torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
            / 255.0
        ),
    )
    real_dataset.data = real_dataset.data[: args.max_real_samples]
    real_dataset.targets = real_dataset.targets[: args.max_real_samples]
    real_loader = DataLoader(
        real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    real_batches = (images for images, _ in real_loader)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    real_features = extract_inception_features(model, real_batches, device)
    generated_features = extract_inception_features(model, generated_batches, device)
    real_mu, real_sigma = (
        real_features.mean(axis=0),
        np.cov(real_features, rowvar=False),
    )
    generated_mu, generated_sigma = (
        generated_features.mean(axis=0),
        np.cov(generated_features, rowvar=False),
    )
    fid = compute_frechet_distance(real_mu, real_sigma, generated_mu, generated_sigma)

    result = {
        "metric": "FID",
        "feature_extractor": "pytorch-fid InceptionV3 (pt_inception-2015-12-05, pre-logits 2048-d)",
        "real_samples": int(len(real_features)),
        "generated_samples": int(len(generated_features)),
        "fid": float(fid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

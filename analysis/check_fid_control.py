"""Validate the FID pipeline with control distributions."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from torchvision.datasets import CIFAR10

FEATURE_DIM = 2048


def frechet_distance(mu1, sigma1, mu2, sigma2):
    from scipy import linalg

    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    mean_diff = mu1 - mu2
    return float(mean_diff @ mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def features(model, images, device, batch_size=128):
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            out.append(model(batch)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CIFAR10(
        root=str(project_root / "data"),
        train=False,
        download=False,
        transform=lambda image: (
            torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
            / 255.0
        ),
    )
    images = torch.stack([dataset[i][0] for i in range(4000)])
    half = len(images) // 2
    real_a, real_b = (images[:half], images[half:])
    noise = torch.rand(2000, 3, 32, 32)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]]).to(device)
    model.eval()
    fa, fb, fn = (
        features(model, real_a, device),
        features(model, real_b, device),
        features(model, noise, device),
    )
    result = {
        "fid_real_vs_real": frechet_distance(
            fa.mean(0), np.cov(fa, rowvar=False), fb.mean(0), np.cov(fb, rowvar=False)
        ),
        "fid_real_vs_uniform_noise": frechet_distance(
            fa.mean(0), np.cov(fa, rowvar=False), fn.mean(0), np.cov(fn, rowvar=False)
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

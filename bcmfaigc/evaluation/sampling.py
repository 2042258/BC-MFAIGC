"""Generate images from a DDPM checkpoint without selecting an experiment version."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image


def load_ddpm_checkpoint(checkpoint_path, model_type, device):
    """Load the existing state-dict schema using the caller's model class."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = model_type(timesteps=int(checkpoint.get("timesteps", 1000))).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def generate_ddpm_samples(model, num_samples, batch_size, seed, device):
    """Sample an evaluation-mode model, preserving the original RNG sequence."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    samples = []
    remaining = num_samples
    with torch.inference_mode():
        while remaining:
            current_batch_size = min(batch_size, remaining)
            samples.append(model.sample(current_batch_size, device=device).cpu())
            remaining -= current_batch_size
    return torch.cat(samples, dim=0)


def save_generated_samples(samples, output_dir):
    """Write the unchanged tensor and preview formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_dir / "samples.pt")
    save_image(
        samples[: min(64, len(samples))],
        output_dir / "preview.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )


def main(argv=None, *, model_type):
    """CLI adapter; the experiment supplies its original DDPM class."""
    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 samples from a saved federated DDPM checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")

    device = torch.device(args.device)
    model = load_ddpm_checkpoint(args.checkpoint, model_type, device)
    samples = generate_ddpm_samples(
        model, args.num_samples, args.batch_size, args.seed, device
    )
    save_generated_samples(samples, args.output_dir)
    print(f"saved {len(samples)} samples to {args.output_dir}")

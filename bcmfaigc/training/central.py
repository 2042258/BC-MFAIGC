"""Train the centralized DDPM generation-quality reference.

The reference has a different training budget from the federated comparisons;
it is not a matched aggregation baseline or a guaranteed quality upper bound.
A nonzero ``--noise-probability`` retains the historical centralized random
weight-corruption condition, not a node-level federated attack model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_images(data: torch.Tensor) -> torch.Tensor:
    return data.mul(2.0).sub(1.0)


def evaluate_denoising_loss(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, seed: int
) -> float:
    model.eval()
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    total_loss = 0.0
    total_elements = 0
    with torch.no_grad():
        for data, _ in loader:
            x_start = normalize_images(data.to(device))
            timestep = torch.randint(
                0,
                model.timesteps,
                (x_start.shape[0],),
                device=device,
                generator=generator,
            )
            noise = torch.randn(x_start.shape, device=device, generator=generator)
            predicted = model(model.q_sample(x_start, timestep, noise), timestep)
            total_loss += F.mse_loss(predicted, noise, reduction="sum").item()
            total_elements += x_start.numel()
    return total_loss / total_elements


def add_gaussian_weight_noise(model: torch.nn.Module, scale: float) -> None:
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.add_(scale * torch.randn_like(module.weight))


def main(argv=None, *, dataset_type, model_type) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--test-batchsize", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, default=9001)
    parser.add_argument("--noise-probability", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--checkpoint-frequency", type=int, default=20)
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.noise_probability <= 1.0:
        raise ValueError("noise-probability must be in [0, 1]")
    if args.epochs <= 0 or args.batchsize <= 0 or args.test_batchsize <= 0:
        raise ValueError("epochs and batch sizes must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("../logs/central") / f"{timestamp}_{args.run_tag}"
    log_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir()

    args_document = vars(args).copy()
    args_document["accumulation_steps"] = 4
    args_document["optimizer_state_reset_each_epoch"] = True
    args_document["timestamp"] = timestamp
    (log_dir / "args_used.json").write_text(
        json.dumps(args_document, indent=2) + "\n", encoding="utf-8"
    )

    dataset = dataset_type("cifar10", isIID=False)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.train_data), torch.from_numpy(dataset.train_label)
        ),
        batch_size=args.batchsize,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.test_data), torch.from_numpy(dataset.test_label)
        ),
        batch_size=args.test_batchsize,
        shuffle=False,
    )

    model = model_type().to(device)
    records_path = log_dir / "central_training_records.jsonl"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-5
        )
        epoch_loss = 0.0
        batches = 0
        pending_steps = 0
        accumulation_steps = 4
        optimizer.zero_grad()
        for data, _ in train_loader:
            x_start = normalize_images(data.to(device))
            loss = model.training_loss(x_start)
            (loss / accumulation_steps).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                pending_steps = 0
            epoch_loss += loss.item()
            batches += 1
        if pending_steps:
            optimizer.step()
            optimizer.zero_grad()

        injected = False
        if args.noise_probability > 0 and random.random() < args.noise_probability:
            add_gaussian_weight_noise(model, args.noise_scale)
            injected = True

        validation_loss = evaluate_denoising_loss(
            model, test_loader, device, args.validation_seed
        )
        record = {
            "epoch": epoch,
            "training_loss_mean": epoch_loss / batches,
            "validation_loss": validation_loss,
            "noise_injected": injected,
            "noise_probability": args.noise_probability,
            "noise_scale": args.noise_scale,
            "seed": args.seed,
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        if epoch == 1 or epoch % args.checkpoint_frequency == 0 or epoch == args.epochs:
            state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "round": epoch,
                    "model_name": "ddpm",
                    "timesteps": model.timesteps,
                    "state_dict": state_dict,
                    "seed": args.seed,
                    "validation_seed": args.validation_seed,
                    "noise_probability": args.noise_probability,
                    "noise_scale": args.noise_scale,
                },
                checkpoint_dir / f"global_round_{epoch:04d}.pt",
            )

    (log_dir / "time.json").write_text(
        json.dumps(
            {"wall_seconds": time.time() - start_time, "epochs": args.epochs}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved centralized DDPM run to {log_dir}")

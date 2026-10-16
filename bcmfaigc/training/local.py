"""Local DDPM optimization and evaluation, independent of node/ledger state.

Training uses the caller's existing global Torch RNG. Evaluation retains the
original fixed validation generator and model-mode restoration boundaries.
"""

import torch
from torch import optim
import torch.nn.functional as F


def create_local_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def unwrap_parallel_model(model):
    """Unwrap only DataParallel, matching the original supported behavior."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def normalize_diffusion_batch(data):
    return data.mul(2.0).sub(1.0)


def compute_denoising_loss(model, core, x_start, device):
    """Predict diffusion noise using the outer model, including DataParallel."""
    timesteps = torch.randint(0, core.timesteps, (x_start.shape[0],), device=device)
    noise = torch.randn_like(x_start)
    noisy_data = core.q_sample(x_start, timesteps, noise)
    predicted_noise = model(noisy_data, timesteps)
    return F.mse_loss(predicted_noise, noise)


def run_local_training_epoch(
    model, optimizer, train_batches, loss_for_batch, accumulation_steps=4
):
    """Mutate model/optimizer using the original full and trailing accumulation.

    The callback keeps the node's existing per-batch loss override active. No
    averaging or last-partial-group rescaling is introduced.
    """
    model.train()
    optimizer.zero_grad()
    pending_steps = 0
    for data, _ in train_batches:
        loss = loss_for_batch(model, data)
        (loss / accumulation_steps).backward()
        pending_steps += 1
        if pending_steps == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
            pending_steps = 0
    if pending_steps:
        optimizer.step()
        optimizer.zero_grad()


def evaluate_denoising_loss(
    model,
    core,
    test_batches,
    device,
    validation_seed,
    normalize_batch=normalize_diffusion_batch,
):
    """Return per-element MSE, restoring the original root training mode.

    Generator setup intentionally remains before try/finally: its failure leaves
    the model in eval mode, as before. Empty input still produces NaN. This differs
    from centralized reference evaluation, so those implementations are not merged.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device.type)
    generator.manual_seed(validation_seed)
    total_loss = 0.0
    num_elements = 0
    try:
        with torch.no_grad():
            for data, _ in test_batches:
                x_start = normalize_batch(data.to(device))
                timesteps = torch.randint(
                    0,
                    core.timesteps,
                    (x_start.shape[0],),
                    device=device,
                    generator=generator,
                )
                noise = torch.randn(x_start.shape, device=device, generator=generator)
                noisy_data = core.q_sample(x_start, timesteps, noise)
                predicted_noise = model(noisy_data, timesteps)
                total_loss += F.mse_loss(predicted_noise, noise, reduction="sum").item()
                num_elements += x_start.numel()
    finally:
        model.train(was_training)
    return total_loss / num_elements if num_elements else float("nan")

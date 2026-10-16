"""Frozen original model-operation methods for characterization tests only.

Never imported by production code. Method bodies retain the original behavior,
including evaluation-mode restoration boundaries and partial accumulation.
"""

import torch
from torch import optim
import torch.nn.functional as F


class LegacyLocalTraining:
    def _make_optimizer(self, model):
        if self.optimizer_name == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate)
        if self.optimizer_name == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    @staticmethod
    def _ddpm_core(model):
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    @staticmethod
    def _normalize_ddpm_batch(data):
        return data.mul(2.0).sub(1.0)

    def _ddpm_training_loss(self, model, data):
        core = self._ddpm_core(model)
        x_start = self._normalize_ddpm_batch(data.to(self.dev))
        timesteps = torch.randint(
            0,
            core.timesteps,
            (x_start.shape[0],),
            device=self.dev,
        )
        noise = torch.randn_like(x_start)
        noisy_data = core.q_sample(x_start, timesteps, noise)
        predicted_noise = model(noisy_data, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def _run_local_epoch(self, model, optimizer, accumulation_steps=4):
        """Run one epoch with identical accumulation for workers and validators."""
        model.train()
        optimizer.zero_grad()
        pending_steps = 0
        for data, _ in self.train_dl:
            loss = self._ddpm_training_loss(model, data)
            (loss / accumulation_steps).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                pending_steps = 0
        if pending_steps:
            optimizer.step()
            optimizer.zero_grad()

    def _evaluate_ddpm_loss(self, model):
        core = self._ddpm_core(model)
        was_training = model.training
        model.eval()
        generator = torch.Generator(device=self.dev.type)
        generator.manual_seed(self.validation_seed)
        total_loss = 0.0
        num_elements = 0
        try:
            with torch.no_grad():
                for data, _ in self.test_dl:
                    x_start = self._normalize_ddpm_batch(data.to(self.dev))
                    timesteps = torch.randint(
                        0,
                        core.timesteps,
                        (x_start.shape[0],),
                        device=self.dev,
                        generator=generator,
                    )
                    noise = torch.randn(
                        x_start.shape,
                        device=self.dev,
                        generator=generator,
                    )
                    noisy_data = core.q_sample(x_start, timesteps, noise)
                    predicted_noise = model(noisy_data, timesteps)
                    total_loss += F.mse_loss(
                        predicted_noise, noise, reduction="sum"
                    ).item()
                    num_elements += x_start.numel()
        finally:
            model.train(was_training)
        return total_loss / num_elements if num_elements else float("nan")

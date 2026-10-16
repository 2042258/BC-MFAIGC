"""DDPM with a time-conditioned convolutional encoder-decoder.

Registered module/buffer names remain unchanged for existing checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_timesteps=1000):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, t):
        return self.fc(self.embedding(t))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(UNet, self).__init__()
        self.time_embedding_dim = time_embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            # DDPM predicts epsilon, which is not bounded to [-1, 1].
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, time_embedding):
        time_embedding = self.time_mlp(time_embedding)
        x = self.encoder(x)
        x = x + time_embedding.view(
            time_embedding.size(0), time_embedding.size(1), 1, 1
        )
        return self.decoder(x)


class DDPM(nn.Module):
    """A compact DDPM with epsilon prediction and ancestral sampling."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super(DDPM, self).__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")

        self.timesteps = timesteps
        self.time_embedding = TimeEmbedding(embedding_dim=128, num_timesteps=timesteps)
        self.unet = UNet(in_channels=3, out_channels=3, time_embedding_dim=128)

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars),
        )

    @staticmethod
    def _extract(values, timesteps, shape):
        extracted = values.gather(0, timesteps)
        return extracted.view(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def forward(self, x, t):
        return self.unet(x, self.time_embedding(t))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alpha_bars, t, x_start.shape
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def training_loss(self, x_start, t=None, noise=None):
        if t is None:
            t = torch.randint(
                0, self.timesteps, (x_start.shape[0],), device=x_start.device
            )
        if noise is None:
            noise = torch.randn_like(x_start)
        noisy_images = self.q_sample(x_start, t, noise)
        predicted_noise = self(noisy_images, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples, device=None, clip_denoised=True):
        device = device or next(self.parameters()).device
        samples = torch.randn(num_samples, 3, 32, 32, device=device)
        self.eval()
        for step in range(self.timesteps - 1, -1, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            predicted_noise = self(samples, t)
            alpha_bar_t = self._extract(self.alpha_bars, t, samples.shape)
            sqrt_one_minus_alpha_bar_t = self._extract(
                self.sqrt_one_minus_alpha_bars, t, samples.shape
            )
            # Predict x_0, clip the predicted clean image, then use the
            # closed-form posterior mean for the ancestral transition.
            predicted_x0 = (
                samples - sqrt_one_minus_alpha_bar_t * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            if clip_denoised:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
            posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, samples.shape)
                * predicted_x0
                + self._extract(self.posterior_mean_coef2, t, samples.shape) * samples
            )
            if step > 0:
                variance = self._extract(self.posterior_variance, t, samples.shape)
                samples = posterior_mean + torch.sqrt(variance) * torch.randn_like(
                    samples
                )
            else:
                samples = posterior_mean
        return samples


# UNet remains the legacy pickle symbol; the public name describes its role.
DenoisingNetwork = UNet

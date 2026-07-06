"""Train a small convolutional VAE on MNIST and dump two figures used in
the slides:

- ``slides/img/05-geradores/vae-samples.png``: 8x8 grid of samples drawn
  from a standard normal prior and passed through the decoder.
- ``slides/img/05-geradores/vae-interpolation.png``: linear interpolation
  in the latent space between two test images of different classes.

Usage (from the repo root)::

    python scripts/train_vae_mnist.py --epochs 15

The script defaults to GPU if available, falls back to CPU. The model is
small enough that 10–20 epochs on CPU are tolerable. Reproducible via the
``--seed`` flag.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "slides" / "img" / "05-geradores"
LATENT_DIM = 16


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decode(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train(model: VAE, loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / len(loader.dataset)
        print(f"epoch {epoch:3d}/{epochs}  loss/sample={avg:.3f}")


def make_sample_grid(model: VAE, device: torch.device, out_path: Path, n: int = 64) -> None:
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        samples = model.decode(z).cpu()
    save_image(samples, out_path, nrow=8, padding=2, pad_value=1.0)
    print(f"wrote {out_path}")


def make_interpolation(model: VAE, test_set, device: torch.device, out_path: Path, n_steps: int = 10) -> None:
    """Interpolate between one image of class 3 and one of class 8."""
    model.eval()
    img_a = next(img for img, lbl in test_set if lbl == 3)
    img_b = next(img for img, lbl in test_set if lbl == 8)
    with torch.no_grad():
        x = torch.stack([img_a, img_b]).to(device)
        mu, _ = model.encode(x)
        z_a, z_b = mu[0], mu[1]
        ts = torch.linspace(0.0, 1.0, n_steps, device=device).unsqueeze(1)
        zs = (1.0 - ts) * z_a + ts * z_b
        decoded = model.decode(zs).cpu()
    save_image(decoded, out_path, nrow=n_steps, padding=2, pad_value=1.0)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "build" / "mnist-data")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = VAE().to(device)
    train(model, loader, device, args.epochs, args.lr)

    make_sample_grid(model, device, OUT_DIR / "vae-samples.png", n=64)
    make_interpolation(model, test_set, device, OUT_DIR / "vae-interpolation.png", n_steps=10)


if __name__ == "__main__":
    main()

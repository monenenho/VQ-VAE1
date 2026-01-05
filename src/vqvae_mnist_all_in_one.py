# vqvae_mnist_all_in_one.py
# ============================================================
# VQ-VAE (MNIST) All-in-One:
# 1) Train
# 2) Save weights (vqvae_mnist.pth)
# 3) Save reconstructions (results/recon_*.png)
# 4) Save 4->9 interpolation (results/interp_4_to_9.png)
# ============================================================

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# -----------------------------
# Model: VQ-VAE
# -----------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        """
        inputs: (B, D, H, W)
        returns:
          vq_loss (scalar),
          quantized (B, D, H, W),
          encoding_indices (B*H*W, 1)
        """
        # (B, D, H, W) -> (B, H, W, D)
        inputs_bhwd = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_bhwd.view(-1, self.embedding_dim)  # (B*H*W, D)

        # distances: ||z||^2 + ||e||^2 - 2 z.e
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W,1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized_bhwd = torch.matmul(encodings, self.embedding.weight).view_as(inputs_bhwd)

        # losses (in BHWD space)
        e_latent_loss = F.mse_loss(quantized_bhwd.detach(), inputs_bhwd)
        q_latent_loss = F.mse_loss(quantized_bhwd, inputs_bhwd.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized_bhwd = inputs_bhwd + (quantized_bhwd - inputs_bhwd).detach()

        # back to (B,D,H,W)
        quantized = quantized_bhwd.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_indices


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, embedding_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, num_hiddens: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        in_channels: int = 1,
        num_hiddens: int = 64,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, num_hiddens, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized, indices = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return vq_loss, x_recon, indices


# -----------------------------
# Utils: save images
# -----------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_tensor_image(t: torch.Tensor, out_path: Path, title: str = ""):
    """
    t: (1,1,H,W) or (1,H,W) or (H,W)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = t.detach().cpu()
    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]

    plt.figure()
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(x, cmap="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grid(images, out_path: Path, title: str = "", ncols: int = 8):
    """
    images: list of tensors, each (1,1,H,W) or (1,H,W) or (H,W)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(ncols * 1.6, nrows * 1.6))
    if title:
        plt.suptitle(title)

    for i, img in enumerate(images):
        x = img.detach().cpu()
        if x.dim() == 4:
            x = x[0, 0]
        elif x.dim() == 3:
            x = x[0]
        plt.subplot(nrows, ncols, i + 1)
        plt.axis("off")
        plt.imshow(x, cmap="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Core: training & generation
# -----------------------------
@torch.no_grad()
def find_one_sample_of_digit(dataset, digit: int, max_tries: int = 20000):
    """
    Return (x, y, index) where y == digit
    """
    # deterministic-ish search
    for i in range(min(len(dataset), max_tries)):
        x, y = dataset[i]
        if int(y) == int(digit):
            return x, y, i
    # fallback random
    for _ in range(20000):
        i = random.randint(0, len(dataset) - 1)
        x, y = dataset[i]
        if int(y) == int(digit):
            return x, y, i
    raise RuntimeError(f"Could not find digit {digit} in dataset.")


@torch.no_grad()
def reconstruct_examples(model: VQVAE, device, dataset, out_dir: Path, digits=(4, 9), count_per_digit: int = 3):
    """
    Save input + recon for a few examples of given digits.
    """
    model.eval()
    saved = 0
    for d in digits:
        n = 0
        # simple scan
        for i in range(len(dataset)):
            x, y = dataset[i]
            if int(y) != int(d):
                continue
            x_b = x.unsqueeze(0).to(device)
            vq_loss, x_recon, _ = model(x_b)

            save_tensor_image(x_b, out_dir / f"recon_digit{d}_{n}_input.png", title=f"input {d}")
            save_tensor_image(x_recon, out_dir / f"recon_digit{d}_{n}_recon.png", title=f"recon {d}")
            n += 1
            saved += 1
            if n >= count_per_digit:
                break
    return saved


@torch.no_grad()
def interpolate_4_to_9(model: VQVAE, device, dataset, out_dir: Path, steps: int = 12):
    """
    Interpolate in *pre-quantization* latent (encoder output z).
    This is the practical way for VQ-VAE to show smooth changes.
    """
    model.eval()

    x4, _, idx4 = find_one_sample_of_digit(dataset, 4)
    x9, _, idx9 = find_one_sample_of_digit(dataset, 9)

    x4_b = x4.unsqueeze(0).to(device)
    x9_b = x9.unsqueeze(0).to(device)

    # encoder output (continuous)
    z4 = model.encoder(x4_b)
    z9 = model.encoder(x9_b)

    # linear interpolation
    imgs = []
    ts = torch.linspace(0.0, 1.0, steps, device=device)
    for t in ts:
        z = (1 - t) * z4 + t * z9
        x_gen = model.decoder(z)  # decode without quantization for smoothness
        imgs.append(x_gen)

    # also save endpoints as reference
    save_tensor_image(x4_b, out_dir / f"interp_input_4_idx{idx4}.png", title="input 4")
    save_tensor_image(x9_b, out_dir / f"interp_input_9_idx{idx9}.png", title="input 9")

    save_grid(imgs, out_dir / "interp_4_to_9.png", title="VQ-VAE (pre-quant) interpolation: 4 -> 9", ncols=steps)
    return idx4, idx9


def train_and_save(
    save_path: Path,
    results_dir: Path,
    batch_size: int = 128,
    num_embeddings: int = 64,
    embedding_dim: int = 16,
    commitment_cost: float = 0.25,
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    ensure_dir(results_dir)

    # dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # Windows安定設定
        pin_memory=True,    # GPUがあるなら有効
    )

    # model
    model = VQVAE(num_embeddings, embedding_dim, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, _ = model(data)
            recon_loss = F.mse_loss(data_recon, data)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            # 進捗（固まって見える対策）
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | batch {batch_idx+1}/{len(train_loader)} | loss {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # save model
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Model saved -> {save_path.resolve()}")

    # generation / visualization
    n_saved = reconstruct_examples(model, device, train_dataset, results_dir, digits=(4, 9), count_per_digit=3)
    print(f"[OK] Saved reconstructions: {n_saved} pairs (input/recon) -> {results_dir.resolve()}")

    idx4, idx9 = interpolate_4_to_9(model, device, train_dataset, results_dir, steps=12)
    print(f"[OK] Saved interpolation 4->9 (using indices {idx4} and {idx9}) -> {(results_dir / 'interp_4_to_9.png').resolve()}")

    print("[DONE] All outputs generated.")


if __name__ == "__main__":
    # 出力先
    save_path = Path("vqvae_mnist.pth")
    results_dir = Path("results")

    # 実行
    train_and_save(
        save_path=save_path,
        results_dir=results_dir,
        batch_size=128,
        num_embeddings=64,
        embedding_dim=16,
        commitment_cost=0.25,
        learning_rate=1e-3,
        num_epochs=10,
    )

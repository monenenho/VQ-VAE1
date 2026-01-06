import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from models import VQVAE
from utils.dataset import get_mnist_dataloader
from utils.logger import log
import matplotlib.pyplot as plt

# 画像保存用
import os

def save_tensor_image(t: torch.Tensor, out_path: Path, title: str = ""):
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

@torch.no_grad()
def find_one_sample_of_digit(dataset, digit: int, max_tries: int = 20000):
    for i in range(min(len(dataset), max_tries)):
        x, y = dataset[i]
        if int(y) == int(digit):
            return x, y, i
    import random
    for _ in range(20000):
        i = random.randint(0, len(dataset) - 1)
        x, y = dataset[i]
        if int(y) == int(digit):
            return x, y, i
    raise RuntimeError(f"Could not find digit {digit} in dataset.")

@torch.no_grad()
def interpolate_4_to_9(model, device, dataset, out_dir: Path, steps: int = 12):
    model.eval()
    x4, _, idx4 = find_one_sample_of_digit(dataset, 4)
    x9, _, idx9 = find_one_sample_of_digit(dataset, 9)
    x4_b = x4.unsqueeze(0).to(device)
    x9_b = x9.unsqueeze(0).to(device)
    z4 = model.encoder(x4_b)
    z9 = model.encoder(x9_b)
    imgs = []
    ts = torch.linspace(0.0, 1.0, steps, device=device)
    for t in ts:
        z = (1 - t) * z4 + t * z9
        x_gen = model.decoder(z)
        imgs.append(x_gen)
    save_tensor_image(x4_b, out_dir / f"interp_input_4_idx{idx4}.png", title="input 4")
    save_tensor_image(x9_b, out_dir / f"interp_input_9_idx{idx9}.png", title="input 9")
    save_grid(imgs, out_dir / "interp_4_to_9.png", title="VQ-VAE (pre-quant) interpolation: 4 -> 9", ncols=steps)
    return idx4, idx9

def main():
    # 設定読み込み
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] device = {device}")
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    # データ
    train_loader, train_dataset = get_mnist_dataloader(batch_size=config["batch_size"], train=True)
    # モデル
    model = VQVAE(
        num_embeddings=config["num_embeddings"],
        embedding_dim=config["embedding_dim"],
        commitment_cost=config["commitment_cost"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # 学習
    model.train()
    epoch_losses = []  # 各エポックの損失を記録
    for epoch in range(config["num_epochs"]):
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
            if (batch_idx + 1) % 50 == 0:
                log(f"  Epoch {epoch+1}/{config['num_epochs']} | batch {batch_idx+1}/{len(train_loader)} | loss {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        log(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

    # 損失グラフの保存
    plt.figure()
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    loss_plot_path = results_dir / 'loss_curve.png'
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    log(f"[OK] Loss curve saved -> {loss_plot_path.resolve()}")
    # 保存
    torch.save(model.state_dict(), config["save_path"])
    log(f"[OK] Model saved -> {Path(config['save_path']).resolve()}")
    # 4→9補間
    idx4, idx9 = interpolate_4_to_9(model, device, train_dataset, results_dir, steps=12)
    log(f"[OK] Saved interpolation 4->9 (using indices {idx4} and {idx9}) -> {(results_dir / 'interp_4_to_9.png').resolve()}")
    log("[DONE] All outputs generated.")

if __name__ == "__main__":
    main()

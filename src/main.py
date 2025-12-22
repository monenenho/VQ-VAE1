import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# model.pyからモデルを読み込む
from model import VQVAE

def train():
    # --- ハイパーパラメータ設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_embeddings = 64
    embedding_dim = 16
    commitment_cost = 0.25
    learning_rate = 1e-3
    num_epochs = 10

    # --- データセット準備 ---
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- モデル・最適化関数の初期化 ---
    model = VQVAE(num_embeddings, embedding_dim, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 学習ループ ---
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            vq_loss, data_recon, _ = model(data)
            recon_loss = F.mse_loss(data_recon, data)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # モデルの保存
    torch.save(model.state_dict(), "vqvae_mnist.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
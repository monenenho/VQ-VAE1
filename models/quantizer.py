import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.register_buffer('_usage_count', torch.zeros(num_embeddings))

    def forward(self, inputs):
        # 入力を平坦化: (B, D, H, W) -> (N, D)
        inputs_bhwd = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_bhwd.view(-1, self.embedding_dim)

        # コサイン類似度用に正規化
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        weight_norm = F.normalize(self.embedding.weight, p=2, dim=1)

        # 距離を計算
        distances = (
            torch.sum(flat_input_norm ** 2, dim=1, keepdim=True)
            + torch.sum(weight_norm ** 2, dim=1)
            - 2 * torch.matmul(flat_input_norm, weight_norm.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Dead Codeの再初期化（学習時のみ）
        if self.training:
            usage = torch.histc(encoding_indices.float(), bins=self.num_embeddings, min=0, max=self.num_embeddings - 1)
            unused_mask = (usage == 0)
            if unused_mask.any():
                n_unused = unused_mask.sum().item()
                rand_indices = torch.randperm(flat_input_norm.size(0))[:n_unused]
                self.embedding.weight.data[unused_mask] = flat_input_norm[rand_indices].to(inputs.device)

        # 量子化
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, weight_norm).view_as(inputs_bhwd)

        # 損失の計算
        input_norm = flat_input_norm.view_as(inputs_bhwd)
        e_loss = F.mse_loss(quantized.detach(), input_norm)
        q_loss = F.mse_loss(quantized, input_norm.detach())
        loss = q_loss + self.commitment_cost * e_loss

        # Straight-Through Estimator (勾配をそのまま流す)
        quantized = input_norm + (quantized - input_norm).detach()

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices
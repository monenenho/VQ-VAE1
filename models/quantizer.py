import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        inputs_bhwd = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_bhwd.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_bhwd = torch.matmul(encodings, self.embedding.weight).view_as(inputs_bhwd)
        e_latent_loss = F.mse_loss(quantized_bhwd.detach(), inputs_bhwd)
        q_latent_loss = F.mse_loss(quantized_bhwd, inputs_bhwd.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized_bhwd = inputs_bhwd + (quantized_bhwd - inputs_bhwd).detach()
        quantized = quantized_bhwd.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_indices

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: (batch, dim, h, w) -> (batch, h, w, dim)
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # 距離計算
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.permute(0, 2, 3, 1).shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs.permute(0, 2, 3, 1))
        q_latent_loss = F.mse_loss(quantized, inputs.permute(0, 2, 3, 1).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs.permute(0, 2, 3, 1) + (quantized - inputs.permute(0, 2, 3, 1)).detach()
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens//2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, in_channels=1, num_hiddens=64):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, num_hiddens, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, indices = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, indices
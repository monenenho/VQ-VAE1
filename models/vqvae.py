import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer

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

from torch import nn
from torch.nn import functional as F
from utils.nn_utils import init_weight
import torch

class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels = input_size, out_channels=hidden_dim, kernel_size = (4, 1), stride = (2, 1), padding=(1,0)),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels = hidden_dim, out_channels=output_size, kernel_size = (4, 1), stride = (2, 1), padding=(1,0)),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main.apply(init_weight)

    def forward(self, inputs):
        # inputs - (B, T, P, Dp_max)
        inputs = inputs.permute(0, 3, 1, 2) # -> (B, Dp_max, T, P)
        outputs = self.main(inputs) # -> (B, C_latent, T_latent, P)
        # print(outputs.shape)
        return outputs


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels = input_size, out_channels=hidden_dim, kernel_size = (4, 1), stride = (2, 1), padding=(1,0)),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels = hidden_dim, out_channels=output_size, kernel_size = (4, 1), stride = (2, 1), padding=(1,0)),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        # inputs - (B, C_latent, T_latent, P)
        outputs = self.main(inputs) # -> (B, Dp_max, T, P)
        outputs = outputs.permute(0, 2, 3, 1) # -> (B, T, P, Dp_max)
        return self.out_net(outputs)


class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE codebook with straight-through estimator.
    Input:  z_e  -> (B, C_latent, T_latent, P)
    Output: z_q  -> (B, C_latent, T_latent, P)
            indices -> (B, T_latent, P)
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(self, z_e):

        """
        z_e: (B, C_latent, T_latent, P)
        """
        B, C, T_latent, P = z_e.shape
        assert C == self.embedding_dim, \
            f"Expected channel dim {self.embedding_dim}, got {C}"
        
        # Move latent channel to last dim
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()   # (B, T_latent, P, C_latent)

        z_flat = z_e_perm.view(-1, C)  # (B*T*P, C_latent)

        # Squared L2 distance to codebook entries
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        indices = torch.argmin(distances, dim=1) # -> (B * T_latent * P,)

        z_q_flat = self.embedding(indices) # -> (B * T_latent * P, C_latent)

        z_q_perm = z_q_flat.view(B, T_latent, P, C) # -> (B, T_latent, P, C_latent)

        # VQ losses
        codebook_loss = F.mse_loss(z_q_perm, z_e_perm.detach())
        commitment_loss = F.mse_loss(z_e_perm, z_q_perm.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_perm = z_e_perm + (z_q_perm - z_e_perm).detach()

        # Reshape to original input shape
        z_q = z_q_perm.permute(0, 3, 1, 2).contiguous()   # -> (B, C_latent, T_latent, P)

        indices = indices.view(B, T_latent, P)
        return z_q, indices, vq_loss, codebook_loss, commitment_loss


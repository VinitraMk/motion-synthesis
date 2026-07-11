from torch import nn
from torch.nn import functional as F
from utils.nn_utils import init_weight
import torch
import numpy as np
from networks.transformer_modules import get_1d_sincos_pos_embed_from_grid, TransformerBlock, CrossAttention

# pick from EricGuo text-to-motion repo
class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size, bias = True)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)

# modified for part-aware editing
class PartMovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(PartMovementConvEncoder, self).__init__()
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


class PartMovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(PartMovementConvDecoder, self).__init__()
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

class MovementEncoder(nn.Module):
    def __init__(self, input_dim,
            hidden_size, # latent_dim
            num_heads = 4, depth = 9, max_seq_len = 10):
        super(MovementEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.motion_cross_attention = CrossAttention(
            dim=hidden_size,
            num_heads=4,
            context_dim=hidden_size
        )
        self.latent_frame_tokens = nn.Parameter(
            torch.randn(1, max_seq_len//4, hidden_size)
        )

        self.embedding = nn.Linear(input_dim, hidden_size)
        self.x_pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_size), requires_grad=False)  # Assuming max sequence length of 120
        self.x_red_pos_embed = nn.Parameter(torch.rand(1, max_seq_len // 4, hidden_size), requires_grad=False) # assuming compressed sequence length of T/4
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, hidden_size * 4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos = np.arange(self.max_seq_len, dtype = np.float32)
        red_pos = np.arange(self.max_seq_len//4, dtype = np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
        red_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, red_pos)
        self.x_pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))
        self.x_red_pos_embed.data.copy_(torch.from_numpy(red_pos_embed).unsqueeze(0))

    
    def masked_mean_pool(self, x, mask):
        # x shape: (B, T, D)
        # mask shape: (B, 1, T, T)
        #valid = (~mask).float()
        diag = mask[:, 0].diagonal(dim1=-2, dim2=-1).float()
        token_mask = (diag == 0.0)
        token_mask = token_mask.unsqueeze(-1)
        x = x * token_mask
        return x.sum(dim=1) / (token_mask.sum(dim=1).clamp(min = 1.0))

    def forward(self, x, key_padding_mask=None):
        # x shape: (B, T, D)
        B = x.shape[0]
        x = self.embedding(x) + self.x_pos_embed
        #if torch.isnan(x).any():
            #print('NaN in x b4')
        for block in self.transformer_blocks:
            x = block(x, input_mask = key_padding_mask)
            #if torch.isnan(x).any():
                #print('NaN in x block')
        #print('block finite: ', torch.isfinite(x).all().item(), "max:", x.abs().max().item())
        #x_global = self.masked_mean_pool(x, key_padding_mask)
        #print('is x_global nan', torch.isnan(x_global).any(), torch.isfinite(x_global).all().item(), "max:", x_global.abs().max().item())
        # get global compressed representation of motion space
        x_reduced = self.latent_frame_tokens.repeat(B, 1, 1) + self.x_red_pos_embed
        x_reduced = self.motion_cross_attention(x_reduced, context = x)

        # mu and logvar stas
        mu = self.fc_mu(x_reduced)
        logvar = self.fc_logvar(x_reduced)
        #print('is mu or logvar nan', torch.isnan(mu).any() or torch.isnan(logvar).any())
        return mu, logvar
    
class MovementDecoder(nn.Module):
    def __init__(self, input_dim, # latent_dim 
            hidden_size, out_dim, num_heads = 4, depth = 9, max_seq_len = 10):
        super(MovementDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.max_seq_len = max_seq_len

        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_size), requires_grad=False)  # Assuming max sequence length of 100
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, hidden_size * 4, context_dim=max_seq_len//4)
            for _ in range(depth)
        ])
        self.motion_seq = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos = np.arange(self.max_seq_len, dtype = np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed))


    def forward(self, z, is_autoregressive = False):
        # x shape: (B, D)
        B = z.shape[0]
        x = self.motion_seq.repeat(B, 1, 1) + self.pos_embed
        #print('is latent in decoder nan', torch.isnan(latent).any())
        #print('ze max and latent max', x.abs().max().item(), latent.abs().max().item())
        #print('x shape after pos enc:', x.shape)
        #print('is x + m0 nan', torch.isnan(x).any())
        for block in self.transformer_blocks:
            if is_autoregressive:
                x = block(x, z)
            else:
                x = block(x)
        x = self.out_proj(x)
        return x
    

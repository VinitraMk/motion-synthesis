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
            latent_size = (1, 512), # latent_dim
            num_heads = 4, depth = 9, max_seq_len = 10):
        super(MovementEncoder, self).__init__()
        T_latent, D_latent = latent_size
        self.input_dim = input_dim
        self.hidden_size = D_latent
        self.t_latent = T_latent
        self.num_heads = num_heads
        self.depth = depth
        self.max_seq_len = max_seq_len
        scale = self.hidden_size ** -0.5
        self.global_motion_tokens = nn.Parameter(
            torch.randn(1, T_latent, input_dim) * scale
        )

        self.embedding = nn.Linear(input_dim, D_latent)
        self.x_pos_embed = nn.Parameter(torch.randn(1, max_seq_len + self.t_latent, D_latent), requires_grad=False)  # Assuming max sequence length of 120
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(D_latent, num_heads, D_latent * 4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(D_latent)
        self.fc_mu = nn.Linear(D_latent, D_latent)
        self.fc_logvar = nn.Linear(D_latent, D_latent)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos = np.arange(self.max_seq_len + self.t_latent, dtype = np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
        self.x_pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

    
    def forward(self, x, key_padding_mask=None):
        # x shape: (B, T, D)
        
        B, T = key_padding_mask.shape
        x = self.embedding(x) + self.x_pos_embed
        
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask = key_padding_mask)
        x = self.norm(x)
        return x
    
class MovementDecoder(nn.Module):
    def __init__(self, input_dim, # latent_dim 
            hidden_size, out_dim, latent_dim = (1, 512), num_heads = 4, depth = 9, max_seq_len = 10):
        super(MovementDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.t_latent = latent_dim[0]

        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_size), requires_grad=False)  # Assuming max sequence length of 100
        self.context_pos_embed = nn.Parameter(torch.randn(1, self.t_latent, hidden_size), requires_grad=False)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, hidden_size * 4, context_dim=hidden_size)
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
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))
        context_pos = np.arange(self.t_latent, dtype = np.float32)
        context_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, context_pos)
        self.context_pos_embed.data.copy_(torch.from_numpy(context_pos_embed).unsqueeze(0))


    def forward(self, z, key_padding_mask = None, attn_mask = None, is_autoregressive = True):
        # x shape: (B, D)
        B = z.shape[0]
        x = self.motion_seq.repeat(B, 1, 1) + self.pos_embed
        z = z + self.context_pos_embed
        for block in self.transformer_blocks:
            if is_autoregressive:
                x = block(x, context = z, key_padding_mask = key_padding_mask, attn_mask = attn_mask)
            else:
                x = block(x, attn_mask = attn_mask)

        x = self.out_proj(x)
        x = self.norm(x)
        return x
    
class MovementSkipEncoder(nn.Module):
    def __init__(self, input_dim,
            latent_size = (1, 512), # latent_dim
            num_heads = 4, depth = 9, max_seq_len = 10):
        super(MovementSkipEncoder, self).__init__()
        assert depth % 2 == 1, "Depth must be an odd number for U-Net type skip connections"
        T_latent, D_latent = latent_size
        num_blocks = depth // 2
        self.input_dim = input_dim
        self.hidden_size = D_latent
        self.t_latent = T_latent
        self.num_heads = num_heads
        self.depth = depth
        self.max_seq_len = max_seq_len
        scale = self.hidden_size ** -0.5
        self.global_motion_tokens = nn.Parameter(
            torch.randn(1, T_latent, input_dim) * scale
        )

        self.embedding = nn.Linear(input_dim, D_latent)
        self.x_pos_embed = nn.Parameter(torch.randn(1, max_seq_len + self.t_latent, D_latent), requires_grad=False)  # Assuming max sequence length of 120
        self.input_blocks = nn.ModuleList([
            TransformerBlock(D_latent, num_heads, D_latent * 4)
            for _ in range(num_blocks)
        ])
        self.middle_block = TransformerBlock(D_latent, num_heads, D_latent * 4)
        self.output_blocks = nn.ModuleList([
            TransformerBlock(D_latent, num_heads, D_latent * 4)
            for _ in range(num_blocks)
        ])
        self.linear_layers = nn.ModuleList([
            nn.Linear(2 * D_latent, D_latent)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(D_latent)
        #self.fc_mu = nn.Linear(D_latent, D_latent)
        #self.fc_logvar = nn.Linear(D_latent, D_latent)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos = np.arange(self.max_seq_len + self.t_latent, dtype = np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
        self.x_pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x shape: (B, T, D)
        B, T = key_padding_mask.shape
        x = self.embedding(x) + self.x_pos_embed
        x_outs = []
        for block in self.input_blocks:
            x = block(x, key_padding_mask = key_padding_mask, attn_mask=attn_mask)
            x_outs.append(x)
        x = self.middle_block(x, key_padding_mask = key_padding_mask, attn_mask=attn_mask)
        for (linear, block) in zip(self.linear_layers, self.output_blocks):
            x = torch.cat([x, x_outs.pop()], dim=-1)
            x = linear(x)
            x = block(x, key_padding_mask = key_padding_mask, attn_mask=attn_mask)
        x = self.norm(x)
        return x
    
class MovementSkipDecoder(nn.Module):
    def __init__(self, input_dim, # latent_dim 
            hidden_size, out_dim, num_heads = 4, depth = 9, max_seq_len = 10):
        super(MovementSkipDecoder, self).__init__()
        assert depth % 2 == 1, "Depth must be an odd number for U-Net type skip connections"
        D_latent = hidden_size
        num_blocks = depth // 2
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.max_seq_len = max_seq_len

        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_size), requires_grad=False)  # Assuming max sequence length of 100
        
        self.motion_seq = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_size)
        )
        self.input_blocks = nn.ModuleList([
            TransformerBlock(D_latent, num_heads, D_latent * 4)
            for _ in range(num_blocks)
        ])
        self.middle_block = TransformerBlock(D_latent, num_heads, D_latent * 4)
        self.output_blocks = nn.ModuleList([
            TransformerBlock(D_latent, num_heads, D_latent * 4)
            for _ in range(num_blocks)
        ])
        self.linear_layers = nn.ModuleList([
            nn.Linear(2 * D_latent, D_latent)
            for _ in range(num_blocks)
        ])
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


    def forward(self, z, key_padding_mask=None, attn_mask = None, is_autoregressive = True):
        # x shape: (B, D)
        B = z.shape[0]
        x = self.motion_seq.repeat(B, 1, 1) + self.pos_embed

        x_outs = []
        for block in self.input_blocks:
            if is_autoregressive:
                x = block(x, context=z, key_padding_mask=key_padding_mask, attn_mask = attn_mask)
            else:
                x = block(x, attn_mask=attn_mask)
            x_outs.append(x)

        if is_autoregressive:
            x = self.middle_block(x, context=z, key_padding_mask=key_padding_mask, attn_mask = attn_mask)
        else:
            x = self.middle_block(x, attn_mask=attn_mask)

        for linear, block in zip(self.linear_layers, self.output_blocks):
            x = torch.cat([x, x_outs.pop()], dim=-1)
            x = linear(x)
            if is_autoregressive:
                x = block(x, context=z, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                x = block(x, attn_mask=attn_mask)

        x = self.out_proj(x)
        x = self.norm(x)
        return x
    

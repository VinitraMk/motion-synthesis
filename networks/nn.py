from torch import nn
from networks.autoencoder_modules import PartMovementConvDecoder, PartMovementConvEncoder, VectorQuantizer
from torch.nn import functional as F
import torch
from timm.models.vision.transforer import PatchEmbed, Attention, Mlp
from networks.transformer_modules import ScalarCondEmbedder, TextEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed 

class MotionVQVAE(nn.Module):
    """
    Encoder/decoder backbone reuses:
      - PartMovementConvEncoder
      - PartMovementConvDecoder

    Flow:
      x (B,T,P,Dp_max) -> encoder -> z_e (B,C_latent,T_latent,P)
                -> quantizer -> z_q
                -> decoder -> x_recon (B,T,P,Dp_max)
    """
    def __init__(
        self,
        input_dim,
        enc_hidden_dim=128,
        dec_hidden_dim=128,
        latent_dim=128,
        num_embeddings=128,
        beta=0.25,
    ):
        super().__init__()

        self.encoder = PartMovementConvEncoder(
            input_size=input_dim, # Dp_max
            hidden_dim=enc_hidden_dim,
            output_size=latent_dim # C_latent
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            beta=beta
        )
        self.decoder = PartMovementConvDecoder(
            input_size=latent_dim, # C_latent
            hidden_dim=dec_hidden_dim,
            output_size=input_dim # Dp_max
        )

    def encode(self, x):
        """
        x: (B, T, P, Dp_max)
        returns:
            z_e: (B, C_latent, T_latent, P)
        """
        z_e = self.encoder(x)
        return z_e

    def quantize(self, z_e):
        return self.quantizer(z_e)

    def decode(self, z_q):
        """
        z_q: (B, C_latent, T_latent, P)
        returns:
            x_recon: (B, T, P, Dp_max)
        """
        x_recon = self.decoder(z_q)
        return x_recon

    # accepts x -> (B, T, P, Dp_max)
    def forward(self, x):
        #x_p = x.permute(0, 2, 1, 3) # -> (B, P, T, Dp_max)
        z_e = self.encode(x) # -> (B, C_latent, T_latent, P)
        z_q, indices, vq_loss, codebook_loss, commitment_loss = self.quantize(z_e)
        x_recon = self.decode(z_e) # -> (B, T, P, Dp_max)

        recon_loss = F.l1_loss(x, x_recon)
        loss = recon_loss + vq_loss

        return {
            "x_recon": x_recon,
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

    @torch.no_grad()
    def encode_to_indices(self, x):
        # x: (B, T, P, Dp_max)
        z_e = self.encode(x) # (B, C_latent, T_latent, P)
        _, indices, _, _, _ = self.quantize(z_e)
        return indices # (B, T_latent, P)
    
    @torch.no_grad()
    def decode_from_indices(self, indices):
        """
        indices: (B, T_latent, P)
        """
        B, T_latent, P = indices.shape
        C_latent = self.quantizer.embedding_dim

        z_q_perm = self.quantizer.embedding(indices) # (B, T_latent, P, C_latent)  

        z_q = z_q_perm.permute(0, 3, 1, 2).contiguous() # (B, C_latent, T_latent, P)

        x_recon = self.decode(z_q)  # (B, T, P, D)
        return x_recon


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        text_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = ScalarCondEmbedder(hidden_size)
        self.d_embedder = ScalarCondEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(text_dim = text_dim, hidden_size=hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.d_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.d_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, d, y):
        """
        Forward pass of DiT.
        x: (N, T, D_latent) tensor of temporal inputs (latent representations of motion)
        t: (N,) tensor of diffusion timesteps
        d: (N,) tensor of diffusion steps
        y: (N, D_text) tensor of text conditions
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, C_latent),
        t = self.t_embedder(t)                   # (N, C_latent)
        d = self.d_embedder(d)                   # (N, C_latent)
        y = self.y_embedder(y, self.training)    # (N, C_latent)
        c = t + d + y                            # (N, C_latent)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, C_latent)
        x = self.final_layer(x, c)                # (N, T, D_latent)
        return x


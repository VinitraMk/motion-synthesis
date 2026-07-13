from torch import nn
from networks.autoencoder_modules import PartMovementConvDecoder, PartMovementConvEncoder, VectorQuantizer, MovementEncoder, MovementDecoder, MovementSkipEncoder, MovementSkipDecoder
from torch.nn import functional as F
import torch
from networks.transformer_modules import ScalarCondEmbedder, TextEmbedder, DiTBlock, FinalLayer, get_1d_sincos_pos_embed_from_grid
import numpy as np
from transformers import AutoTokenizer, AutoModel
from data_utils.motion_processor import recover_from_ric
# motion nn

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
        input_size, # latent dim size
        hidden_size=1152,
        text_dim=384,
        depth=9,
        num_heads=4,
        max_seq_len=10, # output window size of the Dataset class
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.latent_dim = input_size
        self.out_dim = self.latent_dim * 2 if learn_sigma else self.latent_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.x_embedder = nn.Linear(input_size, hidden_size, bias = True)
        self.t_embedder = ScalarCondEmbedder(hidden_size)
        self.d_embedder = ScalarCondEmbedder(hidden_size)
        #self.y_embedder = TextEmbedder(text_dim = text_dim, hidden_size=hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size, bias = True)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size), requires_grad = False
        )

        # condition projection layers
        self.text_pooled_proj = nn.Linear(input_size, hidden_size, bias = True)
        self.text_unpooled_proj = nn.Linear(input_size, hidden_size, bias = True)
        self.cond_fuse = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_dim)
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
        pos = np.arange(self.max_seq_len, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize text projection layer[]
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.constant_(self.text_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.d_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.d_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        '''
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        '''
        # initialize conditional projection biases
        with torch.no_grad():
            for block in self.blocks:
                nn.init.constant_(block.cond_proj.bias, 0)
                nn.init.constant_(block.cond_proj.weight, 0)
                #block.cond_proj.bias[5 * self.hidden_size: 6 * self.hidden_size].fill_(0.5)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        self.text_cond_scale = 2.0

    def encode_text_to_motion_space(self, text_embeddings):
        return self.text_proj(text_embeddings)

    def forward(self, x, t, d, text_pooled_embeddings, text_unpooled_embeddings, text_mask = None):
        """
        Forward pass of DiT.
        x: (N, T, C_latent) tensor of temporal inputs (latent representations of motion)
        t: (N,) tensor of diffusion timesteps
        d: (N,) tensor of diffusion steps
        text_pooled_embeddings: (N, C_text) tensor of pooled text embeddings
        text_unpooled_embeddings: (N, L_text, C_text) tensor of unpooled text embeddings
        text_mask: (N, L_text) tensor of text masks
        """
        #print('x shapes: ', x.size(), self.x_embedder(x).size(), self.pos_embed.size())
        x = self.x_embedder(x) + self.pos_embed  # (N, T, C_latent),
        t = self.t_embedder(t)                   # (N, C_latent)
        d = self.d_embedder(d)                   # (N, C_latent)

        #text_ctx = self.text_proj(text_embeddings)
        #text_global_ctx = text_pooled_embeddings * self.text_cond_scale
        
        #c = t + d + text_pooled_embeddings       # (N, C_latent)
        text_pool_embed = self.text_pooled_proj(text_pooled_embeddings)
        text_unpooled_embed = self.text_unpooled_proj(text_unpooled_embeddings)
        c = torch.cat([t, d, text_pool_embed], dim=-1)
        global_cond_fused = self.cond_fuse(c)
        for block in self.blocks:
            x = block(x, global_cond_fused, text_unpooled_embed, text_mask) # (N, T, C_latent)
        x = self.final_layer(x, global_cond_fused)               # (N, T, D_latent)
        return x

class MotionVAE(nn.Module):
    def __init__(self, dim, hidden_size, max_seq_len = 10, t_latent = 1, num_heads = 6, depth = 9):
        super().__init__()
        #self.encoder = MovementEncoder(
            #input_dim=dim - 4,
            #latent_size=(t_latent, hidden_size),
            #num_heads = num_heads,
            #depth = depth,
            #max_seq_len = max_seq_len
        #)
        #self.decoder = MovementDecoder(
            #input_dim = hidden_size,
            #out_dim = dim,
            #hidden_size=hidden_size,
            #num_heads = num_heads,
            #depth = depth,
            #max_seq_len = max_seq_len
        #)
        self.encoder = MovementSkipEncoder(
            input_dim = dim - 4,
            latent_size= (t_latent, hidden_size),
            num_heads = num_heads,
            depth = depth,
            max_seq_len=max_seq_len
        )
        self.decoder = MovementSkipDecoder(
            input_dim = hidden_size,
            out_dim = dim,
            hidden_size=hidden_size,
            num_heads = num_heads,
            depth = depth,
            max_seq_len = max_seq_len
        )
        self.latent_dim = (t_latent, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        scale = self.latent_dim[1] ** -0.5
        self.global_motion_tokens = nn.Parameter(
            torch.randn(1, self.latent_dim[0], dim - 4) * scale
        )

    def _build_4d_padding_mask(self, key_padding_mask = None):
        # key_padding_mask shape: (B, T) boolean mask
        # Returns a 4D mask of shape (B, 1, T, T) additive mask
        if key_padding_mask is None:
            return None

        valid = key_padding_mask.bool()
        valid = valid[:, :, None] & valid[:, None, :] # [B, T, T]

        key_additive_padmask_4d = torch.zeros_like(valid, dtype=torch.float32)
        key_additive_padmask_4d = key_additive_padmask_4d.masked_fill(~valid, -1e4)
        key_additive_padmask_4d = key_additive_padmask_4d.unsqueeze(1) # [B, 1, T, T]
        return key_additive_padmask_4d
    
    def _get_joint_recon_loss(self, x_recon_jt, x_jt, key_padding_mask=None):
        key_padding_mask_jt = key_padding_mask.unsqueeze(-1).unsqueeze(-1)
        joints_ref_masked = x_jt * key_padding_mask_jt
        joints_recon_masked = x_recon_jt * key_padding_mask_jt
        return F.smooth_l1_loss(joints_recon_masked, joints_ref_masked)
    
    def _get_feature_recon_loss(self, x_recon, x, key_padding_mask=None):
        if key_padding_mask is not None:
            per_feat_loss = F.smooth_l1_loss(x_recon[:, :, :-4], x, reduction = "none", beta = 1.0)
            per_frame_loss = per_feat_loss.mean(dim = -1)
            masked_frame_loss = per_frame_loss * key_padding_mask
            valid_frames_per_sample = key_padding_mask.sum(dim = -1).clamp(min = 1.0) # [B]
            loss_per_sample = masked_frame_loss.sum(dim = 1) / valid_frames_per_sample
            return loss_per_sample.mean()
        else:
            return F.smooth_l1_loss(x_recon, x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, key_padding_mask=None):
        out = self.encoder(x, key_padding_mask=key_padding_mask)
        # extract with just first t_latent token to get global embedding
        dist = out[:, :self.latent_dim[0], :]
        mu = self.fc_mu(dist)
        logvar = self.fc_logvar(dist)

        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar

    def decode(self, z_e, key_padding_mask=None):
        return self.decoder(z_e, input_mask=key_padding_mask)

    def forward(self, x, key_padding_mask=None, beta = 1e-4):
        B, T, D  = x.shape

        # add a CLS token for global embedding
        global_motion_tokens = torch.tile(self.global_motion_tokens, (B, 1, 1))
        x_seq = torch.cat([global_motion_tokens, x], dim=1)
        if key_padding_mask != None:
            aug_mask = torch.cat([torch.ones(B, 1, device = key_padding_mask.device), key_padding_mask], dim = 1)
        else:
            aug_mask = key_padding_mask

        # get a 4d additive mask for timm Attention
        key_padding_mask_4d = self._build_4d_padding_mask(key_padding_mask=aug_mask)

        # get latent and stats from encoder
        z_e, mu, logvar = self.encode(x_seq, key_padding_mask=key_padding_mask_4d)

        # get reconstructed sample 
        x_recon = self.decode(z_e, key_padding_mask=key_padding_mask_4d[:,:,1:, 1:])

        # calculate losses
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss.sum(dim=-1).mean()

        if key_padding_mask is not None:
            #D = x.size(-1)
            #abs_diff = torch.abs(x - x_recon[:,:,:-4])
            #error_per_frame = abs_diff.sum(dim=-1) / D # Sum over the feature dimension - BxT
            #masked_error_per_frame = error_per_frame * key_padding_mask.float()

            #valid_frames_per_sample = key_padding_mask.sum(dim = -1).clamp(min = 1.0)
            #loss_per_sample = masked_error_per_frame.sum(dim = 1) / valid_frames_per_sample # sum over time dimension - Bx1
            #recon_loss = loss_per_sample.mean()
            recon_feat_loss = self._get_feature_recon_loss(x_recon, x, key_padding_mask)
            x_jts = recover_from_ric(x, 22)
            x_recon_jts = recover_from_ric(x_recon, 22)
            recon_joint_loss = self._get_joint_recon_loss(x_recon_jts, x_jts, key_padding_mask)
            #print('input shape', x.shape, x_recon.shape, loss_mask.sum(), recon_l1.sum(), recon_loss)
        else:
            recon_feat_loss = F.smooth_l1_loss(x_recon[:,:,:-4], x)
            x_jts = recover_from_ric(x, 22)
            x_recon_jts = recover_from_ric(x_recon, 22)
            recon_joint_loss = F.smooth_l1_loss(x_recon_jts, x_jts)

        loss = recon_feat_loss + recon_joint_loss + beta * kl_loss

        return {
            "x_recon": x_recon,
            "z_e": z_e,
            "loss": loss,
            "kl_loss": kl_loss,
            "recon_loss": recon_feat_loss + recon_joint_loss,
            "recon_feat_loss": recon_feat_loss,
            "recon_joint_loss": recon_joint_loss,
            "mu": mu,
            "logvar": logvar
        }
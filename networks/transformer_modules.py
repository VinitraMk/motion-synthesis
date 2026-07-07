# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from transformers import AutoTokenizer, AutoModel, CLIPTextModel

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class ScalarCondEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class TextEmbedder(nn.Module):
    def __init__(self, text_dim, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, text_emb):
        return self.proj(text_emb)


# text encoder
class TextTokenEncoder(nn.Module):
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2", device = "cuda"):
        super().__init__()
        if model_name == "clip_text":
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_tokens(self, texts):
        inputs = self.tokenizer(
            texts,
            padding = True,
            truncation = True,
            return_tensors = "pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, outputs.pooler_output, inputs.attention_mask
    

# transformer and attention blocks

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_ff, context_dim = None, dropout = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mh_attn = Attention(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout, qkv_bias = True)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, context_dim=context_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context = None, input_mask = None, context_mask = None):
        #print('is input nan:', torch.isnan(x).any())
        if context != None:
            attn_out = self.mh_attn(x, attn_mask = input_mask)
            #print('context attn_out isnan: ', torch.isnan(attn_out).any())
            attn_out = self.norm1(x + attn_out)
            cross_out = self.cross_attn(attn_out, context, mask = context_mask)
            x = self.norm2(attn_out + cross_out)
        else:
            attn_out = self.mh_attn(x, attn_mask = input_mask)
            #print('attn_out isnan: ', torch.isnan(attn_out).any())
            x = self.norm1(x + attn_out)
            #print('attn_out isnan: ', torch.isnan(x).any())
        ff_out = self.mlp(x)
        #print('is mlp nan: ', torch.isnan(ff_out).any())
        x = self.norm3(x + ff_out)
        #print('is mlp_norm nan: ', torch.isnan(x).any())
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, context_dim = None, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        context_dim = context_dim or dim
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(context_dim, dim, bias = False)
        self.to_v = nn.Linear(context_dim, dim, bias = False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask = None):
        B, N, C = x.shape
        _, M, _ = context.shape
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :].to(dtype = torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, -1e6)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.to_out(out)
    

#################################################################################
#                                 Core DiT Model Block                          #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio = 4.0, context_dim = None):
        super().__init__()
        context_dim = context_dim or hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        #self.attn = CrossAttention(hidden_size, num_heads=num_heads, context_dim=context_dim, dropout=0.1)
        self.self_attn = Attention(hidden_size, num_heads = num_heads, qkv_bias = True, attn_drop = 0.1, proj_drop = 0.1)
        #self.attn_drop = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, context_dim=context_dim, dropout=0.1)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        #approx_gelu = lambda: nn.GELU(approximate="tanh")
        #self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        #self.adaLN_modulation = nn.Sequential(
            #nn.SiLU(),
            #nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        #)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )
        self.cond_proj = nn.Linear(context_dim, 9 * hidden_size, bias=True)
        
        self.last_cross_out = None

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c, text_ctx, text_mask = None):
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = self.cond_proj(c).chunk(9, dim=1)
        h = self.modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(h)

        h_cross = self.modulate(self.norm2(x), shift_ca, scale_ca)
        cross_out = self.cross_attn(h_cross, text_ctx, mask = text_mask)
        x = x + gate_ca.unsqueeze(1) * cross_out
        self.last_cross_out = cross_out

        #x = x + self.mlp(self.norm3(x))
        h = self.modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


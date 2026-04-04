"""Experimental: Standard transformer baselines.

Regular: standard causal attention with QK projections + RoPE.
Token-shifted: K uses previous position's projection (like SWA).
Both use SwiGLU MLP. No conv, no delta memory — pure transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import GatedMLP
from train.configs import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Attention(nn.Module):
    """Multi-head causal attention with RoPE and QK projections."""

    def __init__(self, d_model: int, num_heads: int = 4, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _rope(self, x):
        T = x.shape[2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self._rope(self.q_proj(x).view(B, T, H, d).transpose(1, 2))
        k = self._rope(self.k_proj(x).view(B, T, H, d).transpose(1, 2))
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class TokenShiftAttention(nn.Module):
    """Multi-head causal attention with RoPE. No QK projections — token shift only."""

    def __init__(self, d_model: int, num_heads: int = 4, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _rope(self, x):
        T = x.shape[2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self._rope(x.view(B, T, H, d).transpose(1, 2))
        k = self._rope(F.pad(x[:, :-1], (0, 0, 1, 0)).view(B, T, H, d).transpose(1, 2))
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, token_shift: bool = False):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        if token_shift:
            self.attn = TokenShiftAttention(cfg.d_model, num_heads=cfg.num_heads)
        else:
            self.attn = Attention(cfg.d_model, num_heads=cfg.num_heads)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        normed = self.norm_mlp(x)
        return x + self.mlp(normed, self.mlp.project_up(normed))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, token_shift: bool = False):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(cfg, token_shift=token_shift)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def step(self, token, states=None):
        raise NotImplementedError("Use forward() for now")

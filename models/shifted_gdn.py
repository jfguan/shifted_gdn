"""Experimental: GDN with token shift instead of QK projections.

Same as GatedDeltaNet but replaces separate Q/K projections + convs
with token-shifted normalized hidden states. Tests whether the token
shift prior is sufficient for GDN at scale.

Changes from standard GDN:
- Removed: q_proj, k_proj, q_conv, k_conv (4 projections + 2 convs)
- Added: token shift (write key = previous position, read key = current)
- Kept: v_proj, v_conv, gate, beta, decay, WY correction, output norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.components import CausalConv, GatedMLP
from train.configs import ModelConfig


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class GDNTokenShiftBlock(nn.Module):
    """GDN delta rule memory with token shift instead of QK projections.

    Token shift: write key = l2norm(x_{t-1}), read key = l2norm(x_t).
    Recurrence: S = e^g · S + k ⊗ (β(v - S@k))
    Output:     o = S @ q
    """

    def __init__(self, d_model: int, num_heads: int = 4, d_conv: int = 4, chunk_size: int = 64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        # value projection + conv (kept from GDN)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_conv = CausalConv(d_model, d_conv)

        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        self.alpha_proj = nn.Linear(d_model, num_heads, bias=False)

        dt = torch.empty(num_heads).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
        dt = dt.exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(0, 16).log())
        self.A_log._no_weight_decay = True

        self.norm = RMSNorm(self.head_dim)

    def forward(self, x):
        """x: (B, T, D). returns: (B, T, D)."""
        B, T, D = x.shape
        H, K = self.num_heads, self.head_dim
        C = self.chunk_size

        # token-shifted normalized keys (no projections)
        rk = l2norm(x.view(B, T, H, K)) / (K ** 0.5)  # read key = current
        wk = l2norm(F.pad(x[:, :-1], (0, 0, 1, 0)).view(B, T, H, K))  # write key = previous

        # value with projection + conv (same as GDN)
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, K)
        gate = self.gate_proj(x).view(B, T, H, K)
        beta = self.beta_proj(x).sigmoid()
        decay = -self.A_log.exp().view(1, 1, H) * F.softplus(self.alpha_proj(x) + self.dt_bias)

        # transpose to (B, H, T, K)
        rk, wk, v = [t.transpose(1, 2).float() for t in (rk, wk, v)]

        # scale by beta
        v = v * beta.transpose(1, 2).unsqueeze(-1)
        wk_beta = wk * beta.transpose(1, 2).unsqueeze(-1)

        # pad to chunk boundary
        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            rk = F.pad(rk, (0, 0, 0, pad_len))
            wk = F.pad(wk, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            wk_beta = F.pad(wk_beta, (0, 0, 0, pad_len))
            decay = F.pad(decay, (0, 0, 0, pad_len))

        T_padded = rk.shape[2]

        # reshape into chunks
        rk = rearrange(rk, 'b h (n c) d -> b h n c d', c=C)
        wk = rearrange(wk, 'b h (n c) d -> b h n c d', c=C)
        v = rearrange(v, 'b h (n c) d -> b h n c d', c=C)
        wk_beta = rearrange(wk_beta, 'b h (n c) d -> b h n c d', c=C)
        decay = rearrange(decay, 'b t h -> b h t')
        decay = rearrange(decay, 'b h (n c) -> b h n c', c=C)

        # cumulative decay
        decay = decay.cumsum(-1)
        decay_exp = decay.unsqueeze(-1).exp()
        L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()

        # WY correction
        diag_mask = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=0)
        A = -(wk_beta @ wk.transpose(-1, -2) * L_mask).masked_fill(diag_mask, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
        A = A + torch.eye(C, device=x.device)

        v = A @ v
        wk_cumdecay = A @ (wk_beta * decay_exp)

        # chunk-by-chunk propagation
        num_chunks = T_padded // C
        S = x.new_zeros(B, H, K, K)
        o = torch.zeros_like(v)
        causal_mask = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=1)

        for i in range(num_chunks):
            rk_i, wk_i, v_i = rk[:, :, i], wk[:, :, i], v[:, :, i]
            attn = (rk_i @ wk_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(causal_mask, 0)
            v_prime = wk_cumdecay[:, :, i] @ S
            v_new = v_i - v_prime
            o_inter = (rk_i * decay[:, :, i, :, None].exp()) @ S
            o[:, :, i] = o_inter + attn @ v_new
            decay_weights = (decay[:, :, i, -1, None] - decay[:, :, i]).exp().unsqueeze(-1)
            S = S * decay[:, :, i, -1, None, None].exp() + (wk_i * decay_weights).transpose(-1, -2) @ v_new

        # output
        o = rearrange(o, 'b h n c d -> b (n c) h d')[:, :T]
        o = self.norm(o) * F.silu(gate)
        return self.out_proj(o.reshape(B, T, D))

    def step(self, x, state=None):
        """x: (B, D). returns: (B, D), new_state."""
        B, D = x.shape
        H, K = self.num_heads, self.head_dim

        v_conv_st = state["v_conv"] if state else None
        v_raw, v_conv_st = self.v_conv.step(self.v_proj(x), v_conv_st)

        rk = l2norm(x.view(B, H, K)) / (K ** 0.5)
        v = F.silu(v_raw).view(B, H, K)
        gate = self.gate_proj(x).view(B, H, K)
        beta = self.beta_proj(x).sigmoid().unsqueeze(-1)
        decay = (-self.A_log.exp() * F.softplus(self.alpha_proj(x) + self.dt_bias)).exp().view(B, H, 1, 1)

        S = state["S"] if state else x.new_zeros(B, H, K, K)
        wk = state["wk"] if state else x.new_zeros(B, H, K)

        S = S * decay
        v_err = (v - (S * wk.unsqueeze(-1)).sum(-2)) * beta
        S = S + wk.unsqueeze(-1) * v_err.unsqueeze(-2)
        o = (S * rk.unsqueeze(-1)).sum(-2)

        o = self.norm(o) * F.silu(gate)
        o = self.out_proj(o.reshape(B, D))

        return o, {"S": S, "wk": l2norm(x.view(B, H, K)), "v_conv": v_conv_st}


# -- layer and model --

class GDNTokenShiftLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.delta_norm = RMSNorm(cfg.d_model)
        self.delta = GDNTokenShiftBlock(cfg.d_model, num_heads=cfg.num_heads, d_conv=cfg.d_conv, chunk_size=cfg.chunk_size)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)

    def forward(self, x):
        x = x + self.delta(self.delta_norm(x))
        normed = self.mlp_norm(x)
        return x + self.mlp(normed, self.mlp.project_up(normed))

    def step(self, x, state=None):
        out, delta_st = self.delta.step(self.delta_norm(x), state)
        x = x + out
        normed = self.mlp_norm(x)
        return x + self.mlp(normed, self.mlp.project_up(normed)), delta_st


class GDNTokenShift(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([GDNTokenShiftLayer(cfg) for _ in range(cfg.n_layers)])
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
        x = self.embedding(token).squeeze(1)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states

"""Gated DeltaNet — pure PyTorch, no Triton.

Delta rule memory with input-dependent gating and error-corrective writes.
Chunkwise parallel for training, sequential for inference.

Based on: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025)
Reference: NVlabs/GatedDeltaNet, fla naive_chunk_gated_delta_rule
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


class GatedDeltaNetBlock(nn.Module):
    """Multi-head delta rule memory with input-dependent decay and beta gating.

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

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # short convs on q, k, v (local mixing before delta rule)
        self.q_conv = CausalConv(d_model, d_conv)
        self.k_conv = CausalConv(d_model, d_conv)
        self.v_conv = CausalConv(d_model, d_conv)

        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        self.alpha_proj = nn.Linear(d_model, num_heads, bias=False)

        # Mamba-style dt initialization: inverse-softplus of log-uniform in [0.001, 0.1]
        dt = torch.empty(num_heads).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
        dt = dt.exp()
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(0, 16).log())
        self.A_log._no_weight_decay = True

        self.norm = RMSNorm(self.head_dim)

    def forward(self, x):
        """Chunkwise parallel form.

        x: (B, T, D).
        returns: (B, T, D).
        """
        B, T, D = x.shape
        H, K = self.num_heads, self.head_dim
        C = self.chunk_size

        q = l2norm(F.silu(self.q_conv(self.q_proj(x))).view(B, T, H, K)) / (K ** 0.5)
        k = l2norm(F.silu(self.k_conv(self.k_proj(x))).view(B, T, H, K))
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, K)
        gate = self.gate_proj(x).view(B, T, H, K)
        beta = self.beta_proj(x).sigmoid()
        decay = -self.A_log.exp().view(1, 1, H) * F.softplus(self.alpha_proj(x) + self.dt_bias)

        # Transpose to (B, H, T, K) for matmuls
        q, k, v = [t.transpose(1, 2).float() for t in (q, k, v)]

        # Scale v and k by beta
        v = v * beta.transpose(1, 2).unsqueeze(-1)
        k_beta = k * beta.transpose(1, 2).unsqueeze(-1)

        # Pad to chunk boundary
        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            k_beta = F.pad(k_beta, (0, 0, 0, pad_len))
            decay = F.pad(decay, (0, 0, 0, pad_len))

        T_padded = q.shape[2]

        # Reshape into chunks: (B, H, num_chunks, C, K)
        q = rearrange(q, 'b h (n c) d -> b h n c d', c=C)
        k = rearrange(k, 'b h (n c) d -> b h n c d', c=C)
        v = rearrange(v, 'b h (n c) d -> b h n c d', c=C)
        k_beta = rearrange(k_beta, 'b h (n c) d -> b h n c d', c=C)
        decay = rearrange(decay, 'b t h -> b h t')
        decay = rearrange(decay, 'b h (n c) -> b h n c', c=C)

        # Cumulative decay within each chunk
        decay = decay.cumsum(-1)
        decay_exp = decay.unsqueeze(-1).exp()

        # Intra-chunk causal decay mask: L[i,j] = exp(decay[i] - decay[j]) for i >= j
        L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()

        # WY pre-processing: solve (I + A) to get corrected v and k_beta
        # A = -(k_beta @ k^T) * L_mask, strictly lower triangular
        diag_mask = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=0)
        A = -(k_beta @ k.transpose(-1, -2) * L_mask).masked_fill(diag_mask, 0)

        # (I + A)^{-1} via forward substitution
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
        A = A + torch.eye(C, device=x.device)

        # Corrected values and keys
        v = A @ v
        k_cumdecay = A @ (k_beta * decay_exp)

        # Chunk-by-chunk state propagation
        num_chunks = T_padded // C
        S = x.new_zeros(B, H, K, K)
        o = torch.zeros_like(v)
        causal_mask = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=1)

        for i in range(num_chunks):
            q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]

            # Intra-chunk attention
            attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(causal_mask, 0)

            # Inter-chunk: subtract what S already knows
            v_prime = k_cumdecay[:, :, i] @ S
            v_new = v_i - v_prime

            # Output: inter-chunk read + intra-chunk attention
            o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
            o[:, :, i] = o_inter + attn @ v_new

            # Advance state
            decay_weights = (decay[:, :, i, -1, None] - decay[:, :, i]).exp().unsqueeze(-1)
            S = S * decay[:, :, i, -1, None, None].exp() + (k_i * decay_weights).transpose(-1, -2) @ v_new

        # Unpad, reshape, apply output gate and projection
        o = rearrange(o, 'b h n c d -> b (n c) h d')[:, :T]
        o = self.norm(o) * F.silu(gate)
        o = self.out_proj(o.reshape(B, T, D))
        return o

    def step(self, x, state=None):
        """Recurrent form for inference.

        x: (B, D).
        returns: (B, D), new_state.
        """
        B, D = x.shape
        H, K = self.num_heads, self.head_dim

        q_conv_st = state["q_conv"] if state else None
        k_conv_st = state["k_conv"] if state else None
        v_conv_st = state["v_conv"] if state else None

        q_raw, q_conv_st = self.q_conv.step(self.q_proj(x), q_conv_st)
        k_raw, k_conv_st = self.k_conv.step(self.k_proj(x), k_conv_st)
        v_raw, v_conv_st = self.v_conv.step(self.v_proj(x), v_conv_st)

        q = l2norm(F.silu(q_raw).view(B, H, K)) / (K ** 0.5)
        k = l2norm(F.silu(k_raw).view(B, H, K))
        v = F.silu(v_raw).view(B, H, K)
        gate = self.gate_proj(x).view(B, H, K)
        beta = self.beta_proj(x).sigmoid().unsqueeze(-1)
        decay = (-self.A_log.exp() * F.softplus(self.alpha_proj(x) + self.dt_bias)).exp().view(B, H, 1, 1)

        S = state["S"] if state else x.new_zeros(B, H, K, K)

        # Decay state
        S = S * decay

        # Delta rule: error-corrective write
        v_err = (v - (S * k.unsqueeze(-1)).sum(-2)) * beta
        S = S + k.unsqueeze(-1) * v_err.unsqueeze(-2)

        # Read
        o = (S * q.unsqueeze(-1)).sum(-2)
        o = self.norm(o) * F.silu(gate)
        o = self.out_proj(o.reshape(B, D))

        return o, {"S": S, "q_conv": q_conv_st, "k_conv": k_conv_st, "v_conv": v_conv_st}


class GatedDeltaNetLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.delta_norm = RMSNorm(cfg.d_model)
        self.delta = GatedDeltaNetBlock(cfg.d_model, num_heads=cfg.num_heads, d_conv=cfg.d_conv, chunk_size=cfg.chunk_size)
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


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([GatedDeltaNetLayer(cfg) for _ in range(cfg.n_layers)])
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

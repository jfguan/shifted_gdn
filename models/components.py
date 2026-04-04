"""Reusable components.

- CausalConv: causal depthwise conv1d for local token mixing
- GatedMLP: SwiGLU gated projections for channel mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    """Causal depthwise conv1d with parallel and recurrent forms."""

    def __init__(self, d: int, d_conv: int = 4):
        super().__init__()
        self.d = d
        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(d, d, d_conv, bias=True, groups=d, padding=d_conv - 1)

    def forward(self, x):
        """(B, L, D) -> (B, L, D)."""
        return self.conv1d(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)

    def step(self, x, state=None):
        """(B, D) -> (B, D), state."""
        if state is None:
            state = x.new_zeros(x.shape[0], self.d, self.d_conv - 1)
        conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
        state = conv_input[:, :, 1:]
        assert self.conv1d.bias is not None
        out = (conv_input * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
        return out, state


class GatedMLP(nn.Module):
    """SwiGLU gated projections: up-project, gate, down-project."""

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def project_up(self, x):
        """(B, *, d_model) -> (B, *, d_inner)."""
        return self.proj(x)

    def forward(self, x, val):
        """Gate and project down. x: original input for gate, val: transformed value in d_inner space."""
        return self.out_proj(F.silu(val) * F.silu(self.gate(x)))



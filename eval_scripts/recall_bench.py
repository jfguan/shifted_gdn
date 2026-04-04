"""Associative recall: GDN (delta rule) vs transformer, distance sweep.

Train once at max distance, then eval at all distances.
Distractors use a dedicated PAD token (distinct from keys/values).
Usage: uv run eval_scripts/recall_bench.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- config --

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
N_PAIRS = 4
N_KEYS = 16
N_VALS = 16
VOCAB = N_KEYS + N_VALS + 1  # keys 0-15, values 16-31, PAD=32
PAD_TOKEN = N_KEYS + N_VALS
TRAIN_DISTANCE = 8
EVAL_DISTANCES = [0, 16, 32, 64, 128, 256, 512, 768, 1024, 1536]
MAX_STEPS = 20000
PATIENCE = 1000
BATCH = 32
LR = 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_PATH = "eval_results/recall_bench.png"


def make_batch(n_pairs, distance):
    """Sequence: k1 v1 k2 v2 ... kN vN PAD*distance kQ vQ

    Returns (input_ids, targets, mask).
    mask=1 at value positions (store phase) and the final answer position.
    """
    seq_len = n_pairs * 2 + distance + 2  # +2 for kQ vQ
    seqs = torch.full((BATCH, seq_len), PAD_TOKEN, dtype=torch.long)
    for b in range(BATCH):
        keys = torch.randperm(N_KEYS)[:n_pairs]
        vals = N_KEYS + torch.randperm(N_VALS)[:n_pairs]
        for i in range(n_pairs):
            seqs[b, 2 * i] = keys[i]
            seqs[b, 2 * i + 1] = vals[i]
        qi = torch.randint(0, n_pairs, (1,)).item()
        seqs[b, -2] = keys[qi]
        seqs[b, -1] = vals[qi]

    # targets = next token, mask = value positions only
    input_ids = seqs[:, :-1]
    targets = seqs[:, 1:]
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, 1::2] = 1.0  # value positions in store phase (positions 1,3,5,...)
    mask[:, -1] = 1.0  # final answer
    # zero out mask in PAD region
    store_end = n_pairs * 2
    mask[:, store_end:-1] = 0.0

    return input_ids.to(DEVICE), targets.to(DEVICE), mask.to(DEVICE)


# -- blocks --


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)
            * self.weight
        )


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class CausalConv1d(nn.Module):
    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, bias=True, groups=d, padding=k - 1)
    def forward(self, x):
        return self.conv(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)


class DeltaTSBlock(nn.Module):
    """Delta rule with token shift — no QK projections."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim
        rk = F.normalize(x.view(B, T, H, d), dim=-1)
        wk = F.pad(rk[:, :-1], (0, 0, 0, 0, 1, 0))
        v = self.v_proj(x).view(B, T, H, d)
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)
        beta = self.beta_proj(x).sigmoid().view(B, T, H, 1)
        S = x.new_zeros(B, H, d, d)
        out = []
        for t in range(T):
            err = (v[:, t] - (S * wk[:, t].unsqueeze(-1)).sum(-2)) * beta[:, t]
            S = S + wk[:, t].unsqueeze(-1) * err.unsqueeze(-2)
            out.append((S * rk[:, t].unsqueeze(-1)).sum(-2) * gate[:, t])
        return self.out_proj(torch.stack(out, 1).reshape(B, T, D))


class GDNBlock(nn.Module):
    """GDN: QK projections + short convs + SiLU + delta rule."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_conv = CausalConv1d(d_model)
        self.k_conv = CausalConv1d(d_model)
        self.v_conv = CausalConv1d(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        self.norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim
        q = l2norm(F.silu(self.q_conv(self.q_proj(x))).view(B, T, H, d)) / (d ** 0.5)
        k = l2norm(F.silu(self.k_conv(self.k_proj(x))).view(B, T, H, d))
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, d)
        gate = self.gate_proj(x).view(B, T, H, d)
        beta = self.beta_proj(x).sigmoid().view(B, T, H, 1)
        S = x.new_zeros(B, H, d, d)
        out = []
        for t in range(T):
            err = (v[:, t] - (S * k[:, t].unsqueeze(-1)).sum(-2)) * beta[:, t]
            S = S + k[:, t].unsqueeze(-1) * err.unsqueeze(-2)
            out.append((S * q[:, t].unsqueeze(-1)).sum(-2))
        o = torch.stack(out, 1)
        o = self.norm(o) * F.silu(gate)
        return self.out_proj(o.reshape(B, T, D))


class AttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        angles = torch.outer(torch.arange(max_len).float(), freqs)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _rope(self, x):
        T = x.shape[2]
        c, s = self.cos[:T][None, None], self.sin[:T][None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim
        q = self._rope(self.q_proj(x).view(B, T, H, d).transpose(1, 2))
        k = self._rope(self.k_proj(x).view(B, T, H, d).transpose(1, 2))
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        return self.out_proj(
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            .transpose(1, 2)
            .reshape(B, T, D)
        )


class Model(nn.Module):
    def __init__(self, block_type):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        Block = {"delta_ts": DeltaTSBlock, "gdn": GDNBlock, "transformer": AttnBlock}[block_type]
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        RMSNorm(D_MODEL),
                        Block(D_MODEL, N_HEADS),
                        RMSNorm(D_MODEL),
                        nn.Sequential(
                            nn.Linear(D_MODEL, D_MODEL * 2, bias=False),
                            nn.SiLU(),
                            nn.Linear(D_MODEL * 2, D_MODEL, bias=False),
                        ),
                    ]
                )
                for _ in range(N_LAYERS)
            ]
        )
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x):
        x = self.emb(x)
        for n1, block, n2, mlp in self.layers:
            x = x + block(n1(x))
            x = x + mlp(n2(x))
        return self.head(self.norm(x))


# -- train once, eval at all distances --


def train(model):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best, wait = float("inf"), 0
    for step in range(MAX_STEPS):
        dist = torch.randint(0, TRAIN_DISTANCE + 1, (1,)).item()
        input_ids, targets, mask = make_batch(N_PAIRS, dist)
        logits = model(input_ids)
        per_token = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none"
        )
        loss = (per_token.view_as(mask) * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            with torch.no_grad():
                preds = logits.argmax(-1)
                recall_acc = ((preds == targets) * mask)[
                    :, -1
                ].mean()  # accuracy on final answer
            print(
                f"  step {step}: loss={loss.item():.3f} recall={recall_acc.item():.1%}"
            )
        if loss.item() < best - 0.01:
            best, wait = loss.item(), 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    print(f"  converged at step {step}")


def evaluate(model):
    model.eval()
    results = []
    with torch.no_grad():
        for dist in EVAL_DISTANCES:
            correct, total = 0, 0
            for _ in range(50):
                input_ids, targets, mask = make_batch(N_PAIRS, dist)
                preds = model(input_ids).argmax(-1)
                correct += ((preds == targets) * mask)[:, -1].sum().item()
                total += BATCH
            acc = correct / total
            results.append(acc)
            print(f"  distance={dist}: {acc:.1%}")
    return results


def main():
    import os

    os.makedirs("eval_results", exist_ok=True)
    results = {}
    for name in ["gdn", "transformer"]:
        ckpt_path = f"eval_results/recall_{name}.pt"
        print(f"\n=== {name} ===")
        model = Model(name).to(DEVICE)
        print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K")
        if os.path.exists(ckpt_path):
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            )
            print(f"  loaded {ckpt_path}")
        else:
            train(model)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}")
        results[name] = evaluate(model)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(EVAL_DISTANCES, results["delta_ts"], "o-", label="Delta (token shift)", color="#ff7f0e")
    ax.plot(EVAL_DISTANCES, results["gdn"], "^-", label="GDN (delta + QK + conv)", color="#2ca02c")
    ax.plot(EVAL_DISTANCES, results["transformer"], "s-", label="Transformer (softmax + QK)", color="#1f77b4")
    ax.set(
        xlabel="Distance (PAD tokens)",
        ylabel="Recall Accuracy",
        title=f"Associative Recall: {N_PAIRS} pairs, d={D_MODEL}",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()

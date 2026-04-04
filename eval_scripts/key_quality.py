"""Key quality: rank, cosine similarity, condition number per layer.

Usage: uv run eval_scripts/key_quality.py
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train.configs import GDN_100M, SHIFTED_GDN_100M
from dataclasses import replace
from models import build_model
from data.loader import DatasetName
from data import load_dataset, DataLoader

OUT_PATH = "eval_results/key_quality.png"
MODELS = [
    ("Gated Delta Net 105M", GDN_100M, "checkpoints/gdn_100M_the_stack.pt", "#d62728"),
    ("Shifted Key Gated Delta Net 86M", SHIFTED_GDN_100M, "checkpoints/shifted_gdn_100M_the_stack.pt", "#1f77b4"),
]


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def main():
    dataset = load_dataset(DatasetName.THE_STACK)
    loader = DataLoader(dataset.val, batch_size=1, seq_len=2048)
    x, _ = loader.batch()

    all_results = {}
    for name, cfg_base, ckpt_path, color in MODELS:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = replace(cfg_base, vocab_size=ckpt["model_config"].vocab_size)
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()

        keys = {}
        def hook(li):
            def fn(module, inp, out):
                xi = inp[0]
                H, K = module.num_heads, module.head_dim
                if hasattr(module, "k_proj"):
                    k = l2norm(F.silu(module.k_conv(module.k_proj(xi))).view(1, -1, H, K))
                else:
                    k = l2norm(F.pad(xi[:, :-1], (0, 0, 1, 0)).view(1, -1, H, K))
                keys[li] = k.detach().squeeze(0)
            return fn

        for i, layer in enumerate(model.layers):
            layer.delta.register_forward_hook(hook(i))
        with torch.no_grad():
            model(x)

        layers = sorted(keys)
        ranks, conds, coss = [], [], []
        for li in layers:
            kv = keys[li]
            T, H, K = kv.shape
            r, c, co = [], [], []
            for h in range(H):
                s = torch.linalg.svdvals(kv[:, h].float())
                r.append((s > s[0] * 0.01).sum().item())
                c.append((s[0] / s[-1].clamp(min=1e-8)).item())
                idx = torch.randint(0, T, (500, 2))
                co.append(F.cosine_similarity(kv[idx[:, 0], h], kv[idx[:, 1], h], dim=-1).abs().mean().item())
            ranks.append(sum(r) / H)
            conds.append(sum(c) / H)
            coss.append(sum(co) / H)

        all_results[name] = {"layers": layers, "rank": ranks, "cond": conds, "cos": coss, "color": color}

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("rank", "Effective Rank"), ("cos", "Avg Pairwise Cosine"), ("cond", "Condition Number")]

    for ax, (key, title) in zip(axes, metrics):
        for name, res in all_results.items():
            ax.plot(res["layers"], res[key], "o-", label=name, color=res["color"])
        ax.set(xlabel="Layer", ylabel=title, title=title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if key == "cond":
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

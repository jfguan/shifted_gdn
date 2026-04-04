"""Graph training loss curves for 100M models."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# OUT_PATH = "eval_results/loss_curves_gdn_18M.png"
# MODELS = [
#     ("histories/gdn_18M_the_stack.jsonl", "Gated Delta Net 17.9M"),
#     ("histories/gdn_ts_18M_the_stack.jsonl", "Shifted Key Gated Delta Net 14.7M"),
# ]

OUT_PATH = "eval_results/loss_curves_gdn_100M.png"
MODELS = [
    ("histories/gdn_100M_the_stack.jsonl", "Gated Delta Net 105.2M"),
    ("histories/gdn_ts_100M_the_stack.jsonl", "Shifted Key Gated Delta Net 86.2M"),
]

# OUT_PATH = "eval_results/loss_curves_transformer_18M.png"
# MODELS = [
#     ("histories/transformer_18M_the_stack.jsonl", "Transformer 21.5M"),
#     ("histories/transformer_ts_18M_the_stack.jsonl", "Shifted Key Transformer 17.3M"),
# ]

COLORS = ["#d62728", "#1f77b4"]


def main():
    fig, ax = plt.subplots(figsize=(12, 6))

    for (hist, name), color in zip(MODELS, COLORS):
        tokens, loss = load_history(hist)
        smooth_t, smooth_l = smooth(tokens, loss)
        ax.plot(smooth_t, smooth_l, "-", color=color, linewidth=1, alpha=0.2)

        # power law fit in log-log space, skip first 20% of training
        mask = smooth_t > (40 if smooth_t.max() > 40 else smooth_t.max() * 0.2)
        log_t, log_l = np.log(smooth_t[mask]), np.log(smooth_l[mask])
        coeffs = np.polyfit(log_t, log_l, 1)
        fit_t = np.linspace(smooth_t.min(), smooth_t.max(), 300)
        fit_l = np.exp(np.polyval(coeffs, np.log(fit_t)))
        slope = coeffs[0]
        latest_loss = fit_l[-1]
        ax.plot(
            fit_t,
            fit_l,
            "-",
            color=color,
            linewidth=2,
            label=f"{name} (slope={slope:.2f}, loss={latest_loss:.2f})",
        )

    ax.set(
        xlabel="Tokens (M)",
        ylabel="Train Loss (nats)",
        title="Training Loss (The Stack)",
    )
    ax.set_ylim(0.9, 2.75)
    # ax.set_ylim(0.9, 6)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


SMOOTH_WINDOW = 500


def load_history(path):
    entries = [json.loads(l) for l in open(path) if "train_loss" in l]
    return (
        np.array([e["tokens"] / 1e6 for e in entries]),
        np.array([e["train_loss"] for e in entries]),
    )


def smooth(x, y):
    w = min(SMOOTH_WINDOW, len(y))
    k = np.ones(w) / w
    return x[w - 1 :], np.convolve(y, k, mode="valid")


if __name__ == "__main__":
    main()

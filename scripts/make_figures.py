"""Render public graphics exclusively from measured, hash-verified aggregate results."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cdfv2.cli import verify
from src.common import read_json


def main():
    verify()
    r = read_json(ROOT / "results/published/sealed.json")["methods"]
    labels = ["Original", "Naive", "Contextualized", "LongLLMLingua"]
    values = list(r.values())
    x = np.arange(4)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for offset, key, label, color in [
        (-0.18, "strict_accuracy", "Strict accuracy", "#2864a5"),
        (0.18, "macro_f1", "Macro-F1", "#55a99c"),
    ]:
        bars = ax.bar(x + offset, [100 * v[key] for v in values], 0.35, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Percent (%)",
        ylim=(0, 85),
        title="Same Qwen3 judge · 100 contracts / 1,700 hypotheses per method",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        labels,
        [100 * v["decision_flip_rate"] for v in values],
        color=["#9bacb9", "#dc8745", "#55a99c", "#2864a5"],
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set(
        ylabel="Valid-pair disagreement (%)",
        ylim=(0, 57),
        title="Decision flip relative to Original (not necessarily an error)",
    )
    fig.tight_layout()
    fig.savefig(out / "decision_flip.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, title, ylabel in [
        (
            axes[0],
            [v["compression_ratio"]["mean"] * 100 for v in values],
            "Realized retained context",
            "Tokens / original tokens (%)",
        ),
        (
            axes[1],
            [v["total_latency_seconds"] / 60 for v in values],
            "Preparation + all judgments",
            "Summed item latency (minutes)",
        ),
    ]:
        bars = ax.bar(labels, vals, color=["#9bacb9", "#dc8745", "#55a99c", "#2864a5"])
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(vals) * 1.2))
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle(
        "Unequal budgets; LongLLMLingua compresses separately for 17 hypotheses", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out / "compression_latency.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualise audit results 

Companion to plot_audit_results.py (which plots one chart per model). This script
averages each provider's Exact Match Rate across every model it appears in.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from plot_audit_results import AUDIT_DIR, PASS_THRESHOLD, bar_color, load_results

OUTPUT_DIR = Path(__file__).parent / "audit_plots"


def _model_provider_emrs(data: dict) -> dict[str, float]:
    """Return {provider_name: emr} for providers with a valid float EMR."""
    out: dict[str, float] = {}
    for name, result in data.get("providers", {}).items():
        if isinstance(result, dict):
            emr = result.get("exact_match_rate")
            if isinstance(emr, float):
                out[name] = emr
    return out


def aggregate(results: dict[str, dict]) -> tuple[dict[str, list[float]], list[str], list[str]]:
    """Average each provider's EMR across qualifying models.

    A model qualifies if at least one of its providers has EMR >= PASS_THRESHOLD.
    Returns (provider -> [emrs], kept_models, skipped_models).
    """
    provider_emrs: dict[str, list[float]] = defaultdict(list)
    kept, skipped = [], []
    for model, data in results.items():
        emrs = _model_provider_emrs(data)
        if not emrs or max(emrs.values()) < PASS_THRESHOLD:
            skipped.append(model)
            continue
        kept.append(model)
        for name, emr in emrs.items():
            provider_emrs[name].append(emr)
    return provider_emrs, sorted(kept), sorted(skipped)


def plot_average(provider_emrs: dict[str, list[float]], n_models: int, output_dir: Path) -> None:
    # Sort providers by mean EMR ascending so the best lands at the top of the chart.
    order = sorted(provider_emrs.items(), key=lambda kv: np.mean(kv[1]))
    providers = [name for name, _ in order]
    means = [float(np.mean(v)) for _, v in order]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for _, v in order]
    counts = [len(v) for _, v in order]

    n = len(providers)
    fig_height = max(6, n * 0.3 + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    y = np.arange(n)
    colors = [bar_color(m, "ok") for m in means]
    bars = ax.barh(y, means, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Std-dev error bars (only meaningful when a provider spans >1 model).
    ax.errorbar(
        means, y, xerr=stds, fmt="none", ecolor="#444444",
        elinewidth=1, capsize=3, zorder=5,
    )

    ax.axvline(
        PASS_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.2,
        label=f"{PASS_THRESHOLD:.0%} threshold", zorder=4,
    )

    for bar, mean, count in zip(bars, means, counts):
        yc = bar.get_y() + bar.get_height() / 2
        ax.text(
            mean + 0.005, yc,
            f"{mean:.1%}",
            ha="left", va="center", fontsize=7.5, fontweight="bold",
        )
        ax.text(
            0.005, yc,
            f"n={count}",
            ha="left", va="center", fontsize=6.5, color="#ffffff",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(providers, fontsize=8)
    ax.set_xlabel("Mean Exact Match Rate", fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        f"Average EMR per Provider (across {n_models} qualifying models)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label=f"Pass (≥{PASS_THRESHOLD:.0%})"),
        mpatches.Patch(color="#f39c12", label="Marginal (90–95%)"),
        mpatches.Patch(color="#e74c3c", label="Fail (<90%)"),
    ]
    ax.legend(
        handles=legend_patches + ax.get_legend_handles_labels()[0],
        loc="lower right", fontsize=8, framealpha=0.9,
    )

    fig.tight_layout()
    out_path = output_dir / "average_across_models.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(paths: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if paths:
        import json
        results = {}
        for p in paths:
            f = Path(p)
            try:
                data = json.loads(f.read_text())
                results[data.get("model", f.stem)] = data
            except Exception as e:
                print(f"Could not load {f}: {e}")
    else:
        results = load_results(AUDIT_DIR)

    if not results:
        print(f"No audit results found in {AUDIT_DIR}")
        return

    provider_emrs, kept, skipped = aggregate(results)

    if skipped:
        print(f"Excluded {len(skipped)} model(s) with no provider ≥{PASS_THRESHOLD:.0%}:")
        for m in skipped:
            print(f"  - {m}")
    if not provider_emrs:
        print("No qualifying models to average. Nothing to plot.")
        return

    print(f"\nAveraging {len(provider_emrs)} provider(s) across {len(kept)} qualifying model(s):")
    for m in kept:
        print(f"  + {m}")

    plot_average(provider_emrs, n_models=len(kept), output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])

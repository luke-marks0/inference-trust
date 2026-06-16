#!/usr/bin/env python3
"""Visualise audit results — one bar chart per model."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

AUDIT_DIR = Path(__file__).parent / "audit_results"
OUTPUT_DIR = Path(__file__).parent / "audit_plots"
PASS_THRESHOLD = 0.95


def load_results(audit_dir: Path) -> dict[str, dict]:
    """Load the most recent result file per model."""
    by_model: dict[str, tuple[str, dict]] = {}
    for path in sorted(audit_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        model = data.get("model", path.stem)
        # Keep most recent (files are sorted by name which ends in timestamp)
        by_model[model] = (path.name, data)
    return {model: data for model, (_, data) in by_model.items()}


def extract_providers(data: dict) -> tuple[list[str], list[float | None], list[str]]:
    """Return (providers, emr_values, statuses) sorted by EMR descending."""
    providers, emrs, statuses = [], [], []
    for name, result in data.get("providers", {}).items():
        emr = result.get("exact_match_rate")
        if isinstance(emr, float):
            providers.append(name)
            emrs.append(emr)
            statuses.append("ok")
        elif isinstance(result, dict) and "error" in result:
            err = str(result["error"])
            if "429" in err:
                status = "rate_limited"
            elif "404" in err:
                status = "not_available"
            else:
                status = "error"
            providers.append(name)
            emrs.append(None)
            statuses.append(status)

    # Sort: valid EMRs descending, then errors at end
    order = sorted(
        range(len(providers)),
        key=lambda i: (emrs[i] is None, -(emrs[i] or 0)),
    )
    return (
        [providers[i] for i in order],
        [emrs[i] for i in order],
        [statuses[i] for i in order],
    )


def bar_color(emr: float | None, status: str) -> str:
    if emr is None:
        return "#cccccc"
    if emr >= PASS_THRESHOLD:
        return "#2ecc71"
    if emr >= 0.90:
        return "#f39c12"
    return "#e74c3c"


def plot_model(model: str, data: dict, output_dir: Path) -> None:
    providers, emrs, statuses = extract_providers(data)
    if not providers:
        print(f"  No provider data for {model}, skipping.")
        return

    reference_emr = None
    ref = data.get("reference")
    if isinstance(ref, dict):
        reference_emr = ref.get("exact_match_rate")

    n = len(providers)
    fig_width = max(10, n * 0.7 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(n)
    colors = [bar_color(emr, st) for emr, st in zip(emrs, statuses)]
    bar_heights = [emr if emr is not None else 0.0 for emr in emrs]

    bars = ax.bar(x, bar_heights, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Threshold line
    ax.axhline(PASS_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"{PASS_THRESHOLD:.0%} threshold", zorder=4)

    # Reference line
    if reference_emr is not None:
        ax.axhline(reference_emr, color="#3498db", linestyle=":", linewidth=1.5,
                   label=f"Reference EMR {reference_emr:.1%}", zorder=4)

    # Value labels on bars
    for bar, emr, status in zip(bars, emrs, statuses):
        if emr is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                emr + 0.003,
                f"{emr:.1%}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )
        else:
            label = {"rate_limited": "429", "not_available": "404", "error": "ERR"}.get(status, "?")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.01,
                label,
                ha="center", va="bottom", fontsize=7, color="#666666",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Exact Match Rate", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(model, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label=f"Pass (≥{PASS_THRESHOLD:.0%})"),
        mpatches.Patch(color="#f39c12", label="Marginal (90–95%)"),
        mpatches.Patch(color="#e74c3c", label="Fail (<90%)"),
        mpatches.Patch(color="#cccccc", label="Error / unavailable"),
    ]
    ax.legend(
        handles=legend_patches + ax.get_legend_handles_labels()[0],
        loc="lower right", fontsize=8, framealpha=0.9,
    )

    fig.tight_layout()
    safe_name = model.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def main(paths: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if paths:
        files = [Path(p) for p in paths]
        results = {}
        for f in files:
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

    print(f"Plotting {len(results)} model(s)...")
    for model, data in sorted(results.items()):
        print(f"  {model}")
        plot_model(model, data, OUTPUT_DIR)

    print(f"\nDone. Charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main(sys.argv[1:])

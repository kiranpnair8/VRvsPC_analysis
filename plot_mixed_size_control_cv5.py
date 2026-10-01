#!/usr/bin/env python3
"""Plot the MixedMatched training-size control from saved fold results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("REVE+MLP", "EEGNet")
TARGETS = ("PC", "VR")
CONDITIONS = ("Cross", "MixedMatched", "MixedFull")
DISPLAY_CONDITIONS = ("Cross", "Mixed-Matched", "Mixed-Full")
MODEL_COLORS = {"REVE+MLP": "#3366CC", "EEGNet": "#D9532F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication figures for the mixed-size control ablation."
    )
    parser.add_argument(
        "--results",
        default="./out/cv5/mixed_size_control/mixed_size_control_results.csv",
    )
    parser.add_argument("--out_dir", default="./out/cv5/mixed_size_control")
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def validate_and_summarize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "scenario", "fold", "target_environment", "frr", "eer"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Result CSV is missing columns: {missing}")
    rows = []
    for model in MODELS:
        for target in TARGETS:
            for condition in CONDITIONS:
                scenario = f"{condition} -> {target}"
                subset = frame[
                    (frame["model"] == model) & (frame["scenario"] == scenario)
                ].sort_values("fold")
                if subset["fold"].tolist() != list(range(5)):
                    raise AssertionError(f"Expected folds 0..4 for {model}, {scenario}")
                for metric in ("frr", "eer"):
                    values = pd.to_numeric(subset[metric], errors="raise")
                    if not np.isfinite(values.to_numpy(dtype=float)).all():
                        raise ValueError(f"Non-finite {metric} for {model}, {scenario}")
                    rows.append(
                        {
                            "model": model,
                            "target_environment": target,
                            "condition": condition,
                            "metric": metric.upper(),
                            "mean": float(values.mean()),
                            "sd": float(values.std(ddof=1)),
                            "n_folds": 5,
                        }
                    )
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, out_dir: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=True)
    x = np.arange(len(CONDITIONS), dtype=float)
    width = 0.32
    all_upper = []
    for ax, target, panel in zip(axes, TARGETS, ("(a) PC test", "(b) VR test")):
        for model_index, model in enumerate(MODELS):
            model_rows = summary[
                (summary["model"] == model)
                & (summary["target_environment"] == target)
                & (summary["metric"] == metric)
            ].set_index("condition").loc[list(CONDITIONS)]
            means = 100.0 * model_rows["mean"].to_numpy(dtype=float)
            sds = 100.0 * model_rows["sd"].to_numpy(dtype=float)
            offset = (-0.5 if model_index == 0 else 0.5) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=sds,
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.65,
                capsize=2.5,
                error_kw={"ecolor": "black", "elinewidth": 0.8, "capthick": 0.8},
                label=model,
                zorder=3,
            )
            all_upper.extend((means + sds).tolist())
        ax.set_title(panel, loc="left", fontweight="bold", pad=5)
        ax.set_xticks(x, DISPLAY_CONDITIONS)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"{metric} (%)")
    upper = max(all_upper)
    axes[0].set_ylim(0, max(5.0, np.ceil((upper + 2.0) / 5.0) * 5.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=2, frameon=False, columnspacing=1.4, handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.84, wspace=0.14)
    stem = out_dir / f"mixed_size_control_{metric.lower()}"
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dpi < 400:
        raise ValueError("--dpi must be at least 400")
    results_path = Path(args.results)
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results_path)
    summary = validate_and_summarize(frame)
    summary.to_csv(out_dir / "mixed_size_control_plot_values.csv", index=False)
    for metric in ("FRR", "EER"):
        plot_metric(summary, metric, out_dir, args.dpi)
    print(summary.to_string(index=False))
    print(f"Saved FRR and EER publication figures to {out_dir}")


if __name__ == "__main__":
    main()


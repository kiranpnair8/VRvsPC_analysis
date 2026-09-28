#!/usr/bin/env python3
"""Create publication FAR/FRR bar charts from corrected supervised CV5 results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["REVE+MLP", "REVE+SVM", "REVE+KNN", "REVE+RF", "EEGNet"]
MODEL_LABELS = ["REVE+MLP", "REVE+SVM", "REVE+k-NN", "REVE+RF", "EEGNet"]
SCENARIO_ORDER = ["PC->PC", "VR->VR", "PC->VR", "VR->PC", "Mixed->PC", "Mixed->VR"]
METRICS = ["FAR", "FRR"]

FIGURE_SPECS = {
    "within_environment": {
        "scenarios": ["PC->PC", "VR->VR"],
        "condition_labels": ["PC", "VR"],
    },
    "cross_environment": {
        "scenarios": ["PC->VR", "VR->PC"],
        "condition_labels": ["PC", "VR"],
    },
    "mixed_environment": {
        "scenarios": ["Mixed->PC", "Mixed->VR"],
        "condition_labels": ["PC", "VR"],
    },
}

PC_COLOR = (0.20, 0.40, 0.80)
VR_COLOR = (0.85, 0.33, 0.20)
FIGSIZE = (9.0, 5.0)  # MATLAB used a 900 x 500 pixel figure.
OLD_Y_LIMIT = (0.0, 65.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot corrected supervised CV5 FAR/FRR means with sample-SD error bars."
    )
    parser.add_argument(
        "--summary",
        default="./out/cv5/results/supervised_cv5_summary.csv",
        help="Long-form CV5 summary CSV produced by run_cv5_supervised.py",
    )
    parser.add_argument(
        "--per_fold",
        default=None,
        help=(
            "Per-fold CSV used to independently verify means and ddof=1 SDs. "
            "Defaults to supervised_cv5_per_fold.csv beside --summary."
        ),
    )
    parser.add_argument("--out_dir", default="./figures_cv5")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--input_units",
        choices=["fraction", "percent"],
        default="fraction",
        help="The CV5 pipeline writes fractions; plots are always displayed as percentages.",
    )
    parser.add_argument(
        "--skip_fold_crosscheck",
        action="store_true",
        help="Allow plotting from the summary alone. Not recommended for manuscript figures.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def expected_keys() -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [MODEL_ORDER, SCENARIO_ORDER, METRICS], names=["model", "scenario", "metric"]
    )


def load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CV5 summary CSV not found: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        {"model", "scenario", "metric", "metric_mean", "metric_sd", "n_folds"},
        path,
    )
    df = df[
        df["model"].isin(MODEL_ORDER)
        & df["scenario"].isin(SCENARIO_ORDER)
        & df["metric"].isin(METRICS)
    ].copy()

    duplicate = df.duplicated(["model", "scenario", "metric"], keep=False)
    if duplicate.any():
        rows = df.loc[duplicate, ["model", "scenario", "metric"]]
        raise ValueError(f"Duplicate supervised summary rows found:\n{rows.to_string(index=False)}")

    indexed = df.set_index(["model", "scenario", "metric"])
    missing = expected_keys().difference(indexed.index)
    extra = indexed.index.difference(expected_keys())
    if len(missing) or len(extra):
        raise ValueError(
            "CV5 summary does not contain exactly the expected 5 models x 6 scenarios x 2 metrics. "
            f"Missing={list(missing)}; extra={list(extra)}"
        )
    if len(indexed) != 60:
        raise ValueError(f"Expected 60 summary rows, found {len(indexed)}")

    numeric = indexed[["metric_mean", "metric_sd", "n_folds"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Summary contains missing or non-finite mean, SD, or fold-count values")
    if not (numeric["n_folds"] == 5).all():
        bad = numeric.loc[numeric["n_folds"] != 5, "n_folds"]
        raise ValueError(f"Every plotted result must summarize exactly five folds; found:\n{bad}")
    if (numeric["metric_sd"] < 0).any():
        raise ValueError("Summary contains a negative standard deviation")

    indexed[["metric_mean", "metric_sd", "n_folds"]] = numeric
    return indexed.sort_index()


def crosscheck_per_fold(summary: pd.DataFrame, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Per-fold CV5 CSV not found: {path}. Supply --per_fold or, only when necessary, "
            "use --skip_fold_crosscheck."
        )

    folds = pd.read_csv(path)
    require_columns(folds, {"model", "scenario", "fold", "FAR", "FRR"}, path)
    folds = folds[
        folds["model"].isin(MODEL_ORDER) & folds["scenario"].isin(SCENARIO_ORDER)
    ].copy()

    duplicate = folds.duplicated(["model", "scenario", "fold"], keep=False)
    if duplicate.any():
        rows = folds.loc[duplicate, ["model", "scenario", "fold"]]
        raise ValueError(f"Duplicate per-fold rows found:\n{rows.to_string(index=False)}")

    counts = folds.groupby(["model", "scenario"])["fold"].nunique()
    expected_pairs = pd.MultiIndex.from_product(
        [MODEL_ORDER, SCENARIO_ORDER], names=["model", "scenario"]
    )
    counts = counts.reindex(expected_pairs)
    if counts.isna().any() or not (counts == 5).all():
        raise ValueError(
            "Every model/scenario must have exactly five unique fold rows; found:\n"
            + counts.to_string()
        )

    long = folds.melt(
        id_vars=["model", "scenario", "fold"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    if not np.isfinite(long["value"].to_numpy(dtype=float)).all():
        raise ValueError("Per-fold FAR/FRR values contain missing or non-finite entries")

    recomputed = long.groupby(["model", "scenario", "metric"])["value"].agg(
        metric_mean="mean",
        metric_sd=lambda values: values.std(ddof=1),
    )
    expected = summary.loc[recomputed.index, ["metric_mean", "metric_sd"]]
    delta = (recomputed - expected).abs()
    mismatch = ~np.isclose(
        recomputed.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
    )
    if mismatch.any():
        bad_rows = np.any(mismatch, axis=1)
        details = recomputed.loc[bad_rows].join(
            expected.loc[bad_rows], lsuffix="_recomputed", rsuffix="_summary"
        )
        raise ValueError(
            "Summary does not match means/sample SDs recomputed from five folds (ddof=1):\n"
            + details.to_string()
        )

    max_delta = float(delta.to_numpy(dtype=float).max())
    print(f"Fold cross-check passed: 5 folds per model/scenario, ddof=1, max delta={max_delta:.3g}")


def values_table(summary: pd.DataFrame, scale: float) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for scenario in SCENARIO_ORDER:
            far = summary.loc[(model, scenario, "FAR")]
            frr = summary.loc[(model, scenario, "FRR")]
            rows.append(
                {
                    "Model": model,
                    "Scenario": scenario,
                    "FAR mean (%)": float(far["metric_mean"]) * scale,
                    "FAR SD (%)": float(far["metric_sd"]) * scale,
                    "FRR mean (%)": float(frr["metric_mean"]) * scale,
                    "FRR SD (%)": float(frr["metric_sd"]) * scale,
                }
            )
    return pd.DataFrame(rows)


def figure_arrays(values: pd.DataFrame, scenarios: list[str]) -> tuple[np.ndarray, np.ndarray]:
    indexed = values.set_index(["Model", "Scenario"])
    mean_columns = []
    sd_columns = []
    for scenario in scenarios:
        scenario_rows = indexed.loc[(MODEL_ORDER, scenario), :]
        mean_columns.extend(
            [scenario_rows["FAR mean (%)"].to_numpy(), scenario_rows["FRR mean (%)"].to_numpy()]
        )
        sd_columns.extend(
            [scenario_rows["FAR SD (%)"].to_numpy(), scenario_rows["FRR SD (%)"].to_numpy()]
        )
    return np.column_stack(mean_columns), np.column_stack(sd_columns)


def y_limits(means: np.ndarray, sds: np.ndarray) -> tuple[float, float]:
    low = float(np.min(means - sds))
    high = float(np.max(means + sds))
    lower = OLD_Y_LIMIT[0] if low >= 0 else 5.0 * math.floor((low - 1.0) / 5.0)
    upper = max(OLD_Y_LIMIT[1], 5.0 * math.ceil((high + 2.0) / 5.0))
    return lower, upper


def plot_group(
    values: pd.DataFrame,
    name: str,
    scenarios: list[str],
    condition_labels: list[str],
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    means, sds = figure_arrays(values, scenarios)
    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    colors = [PC_COLOR, PC_COLOR, VR_COLOR, VR_COLOR]
    alphas = [1.0, 0.45, 1.0, 0.45]
    labels = [
        f"FAR ({condition_labels[0]})",
        f"FRR ({condition_labels[0]})",
        f"FAR ({condition_labels[1]})",
        f"FRR ({condition_labels[1]})",
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.linewidth": 1.2,
        }
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for column in range(4):
        ax.bar(
            x + offsets[column],
            means[:, column],
            width=width,
            yerr=sds[:, column],
            label=labels[column],
            color=colors[column],
            alpha=alphas[column],
            edgecolor="black",
            linewidth=1.2,
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
            zorder=3,
        )

    lower, upper = y_limits(means, sds)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x, MODEL_LABELS, rotation=15, ha="right", rotation_mode="anchor")
    ax.grid(True, which="major", axis="both", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.legend(loc="upper left", frameon=True)
    ax.tick_params(width=1.2, labelsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.96)

    if float(np.max(means + sds)) >= upper or float(np.min(means - sds)) <= lower:
        raise RuntimeError(f"Computed y-axis limits would clip an error bar in {name}")

    outputs = [out_dir / f"cv5_{name}_far_frr.pdf", out_dir / f"cv5_{name}_far_frr.png"]
    fig.savefig(outputs[0], format="pdf")
    fig.savefig(outputs[1], format="png", dpi=dpi)
    plt.close(fig)
    print(f"Saved {outputs[0]} and {outputs[1]} (y-limits: {lower:g}, {upper:g})")
    return outputs


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300 for publication output")

    summary_path = Path(args.summary)
    per_fold_path = (
        Path(args.per_fold)
        if args.per_fold
        else summary_path.with_name("supervised_cv5_per_fold.csv")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    if not args.skip_fold_crosscheck:
        crosscheck_per_fold(summary, per_fold_path)

    scale = 100.0 if args.input_units == "fraction" else 1.0
    values = values_table(summary, scale)
    if (values.filter(like="mean").to_numpy(dtype=float) < 0).any():
        raise ValueError("FAR/FRR means must be non-negative")

    audit_path = out_dir / "cv5_far_frr_values_used.csv"
    values.to_csv(audit_path, index=False)
    print("\nExact values used for the plotted bars (mean +/- sample SD across five folds):")
    print(values.to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    print(f"\nValidated 5 models x 6 scenarios x 2 metrics = 60 plotted means with 60 SDs.")
    print(f"Saved numerical audit table: {audit_path}")

    outputs = []
    for name, spec in FIGURE_SPECS.items():
        outputs.extend(
            plot_group(
                values,
                name,
                spec["scenarios"],
                spec["condition_labels"],
                out_dir,
                args.dpi,
            )
        )

    missing_outputs = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"Expected figure files were not created: {missing_outputs}")
    print("All figures generated from corrected CV5 CSV values; no historical arrays were used.")


if __name__ == "__main__":
    main()


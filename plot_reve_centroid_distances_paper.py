#!/usr/bin/env python3
"""Create the paper version of the REVE centroid-distance figure.

This script is deliberately plotting-only. It reads the subject-level values
written by analyze_reve_embeddings.py and does not recompute any distances or
statistical tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


EXPECTED_SUBJECTS = 21
METRICS = (
    ("euclidean", "(a) Euclidean", "Euclidean distance"),
    ("cosine", "(b) Cosine", "Cosine distance"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the paper version of the REVE centroid-distance figure."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("reve_embedding_analysis/subject_level_distances.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reve_embedding_analysis"),
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(frame: pd.DataFrame) -> dict:
    required = {"subject_label"}
    for metric, _, _ in METRICS:
        required.update({f"within_reference_{metric}", f"cross_{metric}"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(frame) != EXPECTED_SUBJECTS:
        raise AssertionError(
            f"Expected {EXPECTED_SUBJECTS} subject rows, found {len(frame)}"
        )
    if frame["subject_label"].nunique() != EXPECTED_SUBJECTS:
        raise AssertionError("subject_label must contain exactly 21 unique subjects")
    if frame["subject_label"].duplicated().any():
        raise AssertionError("Each subject must contribute exactly one paired observation")

    result = {
        "n_subjects": EXPECTED_SUBJECTS,
        "one_pair_per_subject": True,
        "values_recomputed": False,
        "panels": {},
    }
    for metric, _, _ in METRICS:
        within_col = f"within_reference_{metric}"
        cross_col = f"cross_{metric}"
        values = frame[[within_col, cross_col]].to_numpy(dtype=float)
        if values.shape != (EXPECTED_SUBJECTS, 2) or not np.isfinite(values).all():
            raise AssertionError(f"Invalid values for {metric}: shape={values.shape}")
        result["panels"][metric] = {
            "within_column": within_col,
            "cross_column": cross_col,
            "n_pairs": EXPECTED_SUBJECTS,
            "all_pc_vr_greater_than_within": bool(np.all(values[:, 1] > values[:, 0])),
            "n_pc_vr_greater_than_within": int(np.sum(values[:, 1] > values[:, 0])),
            "n_equal": int(np.sum(values[:, 1] == values[:, 0])),
        }
    return result


def plot(frame: pd.DataFrame, out_dir: Path, dpi: int) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    gray = "#737373"
    displacement = "#A23B72"
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))

    for ax, (metric, panel_title, ylabel) in zip(axes, METRICS):
        within = frame[f"within_reference_{metric}"].to_numpy(dtype=float)
        cross = frame[f"cross_{metric}"].to_numpy(dtype=float)

        for within_value, cross_value in zip(within, cross):
            ax.plot(
                [0, 1],
                [within_value, cross_value],
                color="#A6A6A6",
                alpha=0.42,
                linewidth=0.65,
                zorder=1,
            )
        ax.scatter(
            np.zeros(EXPECTED_SUBJECTS), within, s=22, color=gray,
            edgecolors="white", linewidths=0.35, zorder=3,
        )
        ax.scatter(
            np.ones(EXPECTED_SUBJECTS), cross, s=22, color=displacement,
            edgecolors="white", linewidths=0.35, zorder=3,
        )
        ax.set_title(panel_title, loc="left", fontweight="bold", pad=7)
        ax.set_xticks([0, 1], ["Within-env.", "PC--VR"])
        ax.set_xlim(-0.28, 1.28)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", alpha=0.7, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=gray,
            markeredgecolor="white", markeredgewidth=0.35,
            label="Within-environment reference", markersize=5,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="", color=displacement,
            markeredgecolor="white", markeredgewidth=0.35,
            label="PC--VR displacement", markersize=5,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.82, wspace=0.32)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "subject_centroid_distance_comparison_paper"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300 for the paper figure")
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Missing input table: {args.input_csv}")

    input_hash = sha256(args.input_csv)
    frame = pd.read_csv(args.input_csv).sort_values("subject_label").reset_index(drop=True)
    validation = validate(frame)
    pdf_path, png_path = plot(frame, args.out_dir, args.dpi)

    validation.update(
        {
            "input_csv": str(args.input_csv.resolve()),
            "input_csv_sha256": input_hash,
            "pdf": str(pdf_path.resolve()),
            "png": str(png_path.resolve()),
            "png_dpi": args.dpi,
        }
    )
    report_path = args.out_dir / "subject_centroid_distance_comparison_paper_validation.json"
    report_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(json.dumps(validation, indent=2))
    print(f"Saved vector PDF: {pdf_path}")
    print(f"Saved {args.dpi}-DPI PNG: {png_path}")


if __name__ == "__main__":
    main()


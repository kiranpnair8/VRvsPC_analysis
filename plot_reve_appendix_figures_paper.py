#!/usr/bin/env python3
"""Render publication figures from saved REVE PCA and t-SNE coordinates.

This script never fits PCA or t-SNE. It only validates and plots coordinates
previously written by analyze_reve_embeddings.py.
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


EXPECTED_TOTAL = 5040
EXPECTED_SUBJECTS = 21
EXPECTED_PER_ENVIRONMENT = 2520
EXPECTED_PER_SUBJECT = 240
EXPECTED_PER_SUBJECT_ENVIRONMENT = 120

PC_COLOR = "#3366CC"
VR_COLOR = "#D9532F"
ENV_COLORS = {"PC": PC_COLOR, "VR": VR_COLOR}
ENV_MARKERS = {"PC": "o", "VR": "^"}

STYLE = {
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.2,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PCA_GLOBAL_SIZE = 7.0
PCA_GLOBAL_ALPHA = 0.42
PCA_SUBJECT_SIZE = 5.0
PCA_SUBJECT_ALPHA = 0.45
PCA_CENTROID_SIZE = 44.0
TSNE_SIZE = 5.5
TSNE_ALPHA = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final REVE appendix figures from saved coordinates."
    )
    parser.add_argument(
        "--analysis_dir", type=Path, default=Path("reve_embedding_analysis")
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> dict[str, str]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf.resolve()), "png": str(png.resolve())}


def validate_counts(frame: pd.DataFrame, source: str) -> dict:
    required = {
        "row_index", "subject_label", "environment", "environment_epoch",
        "subject_epoch", "outer_test_fold",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing metadata columns: {missing}")

    if len(frame) != EXPECTED_TOTAL:
        raise AssertionError(f"{source}: expected 5040 rows, found {len(frame)}")
    if frame["row_index"].nunique() != EXPECTED_TOTAL:
        raise AssertionError(f"{source}: row_index is not unique")
    if not np.array_equal(
        np.sort(frame["row_index"].to_numpy(dtype=int)), np.arange(EXPECTED_TOTAL)
    ):
        raise AssertionError(f"{source}: row_index must cover 0..5039")

    subjects = np.sort(frame["subject_label"].unique())
    if not np.array_equal(subjects, np.arange(EXPECTED_SUBJECTS)):
        raise AssertionError(
            f"{source}: expected internal subject labels 0..20, found {subjects.tolist()}"
        )
    environment_counts = frame["environment"].value_counts().to_dict()
    if environment_counts != {"PC": 2520, "VR": 2520}:
        raise AssertionError(f"{source}: unexpected environment counts {environment_counts}")

    subject_counts = frame.groupby("subject_label", sort=True).size()
    if not (subject_counts == EXPECTED_PER_SUBJECT).all():
        raise AssertionError(f"{source}: every subject must have 240 rows")
    subject_environment = frame.groupby(
        ["subject_label", "environment"], sort=True
    ).size()
    if not (subject_environment == EXPECTED_PER_SUBJECT_ENVIRONMENT).all():
        raise AssertionError(f"{source}: every subject/environment must have 120 rows")

    return {
        "total": int(len(frame)),
        "PC": int(environment_counts["PC"]),
        "VR": int(environment_counts["VR"]),
        "subjects": int(len(subjects)),
        "per_subject": EXPECTED_PER_SUBJECT,
        "per_subject_environment": EXPECTED_PER_SUBJECT_ENVIRONMENT,
    }


def load_inputs(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    paths = {
        "pca_coordinates": analysis_dir / "pca_coordinates_2d.csv",
        "pca_variance": analysis_dir / "pca_explained_variance.csv",
        "tsne_coordinates": analysis_dir / "tsne_coordinates.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved analysis files: {missing}")

    pca = pd.read_csv(paths["pca_coordinates"]).sort_values("row_index").reset_index(drop=True)
    tsne = pd.read_csv(paths["tsne_coordinates"]).sort_values("row_index").reset_index(drop=True)
    variance = pd.read_csv(paths["pca_variance"])

    for columns, source in [(("PC1", "PC2"), "PCA"), (("tSNE1", "tSNE2"), "t-SNE")]:
        if not set(columns).issubset(pca.columns if source == "PCA" else tsne.columns):
            raise ValueError(f"{source} coordinate columns {columns} are missing")

    pca_counts = validate_counts(pca, "pca_coordinates_2d.csv")
    tsne_counts = validate_counts(tsne, "tsne_coordinates.csv")
    metadata_columns = [
        "row_index", "subject_label", "environment", "environment_epoch",
        "subject_epoch", "outer_test_fold",
    ]
    if not pca[metadata_columns].equals(tsne[metadata_columns]):
        raise AssertionError("PCA and t-SNE rows do not have identical saved metadata")

    if len(variance) < 2 or "explained_variance_ratio" not in variance.columns:
        raise ValueError("pca_explained_variance.csv does not contain PC1/PC2 ratios")
    explained = variance["explained_variance_ratio"].to_numpy(dtype=float)[:2]
    if not np.isfinite(explained).all() or np.any(explained <= 0):
        raise AssertionError(f"Invalid PCA explained variance values: {explained}")

    pca_npz = analysis_dir / "pca_coordinates.npz"
    pca_npz_match = None
    if pca_npz.is_file():
        with np.load(pca_npz, allow_pickle=False) as archive:
            saved_coordinates = archive["coordinates"][:, :2]
            saved_variance = archive["explained_variance_ratio"][:2]
        pca_npz_match = bool(
            np.allclose(saved_coordinates, pca[["PC1", "PC2"]].to_numpy(), rtol=0, atol=1e-6)
            and np.allclose(saved_variance, explained, rtol=0, atol=1e-12)
        )
        if not pca_npz_match:
            raise AssertionError("PCA CSV does not match pca_coordinates.npz")

    validation = {
        "input_files": {name: str(path.resolve()) for name, path in paths.items()},
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "pca_counts": pca_counts,
        "tsne_counts": tsne_counts,
        "metadata_rows_identical": True,
        "pca_coordinates_reused": True,
        "pca_refit": False,
        "pca_npz_match": pca_npz_match,
        "tsne_coordinates_reused": True,
        "tsne_recomputed": False,
        "explained_variance_ratio": {"PC1": float(explained[0]), "PC2": float(explained[1])},
        "subject_labels_modified_in_data": False,
        "display_subject_labels": "internal label + 1",
    }
    return pca, tsne, explained, validation


def variance_labels(explained: np.ndarray) -> tuple[str, str]:
    return (
        f"PC1 ({100.0 * explained[0]:.2f}%)",
        f"PC2 ({100.0 * explained[1]:.2f}%)",
    )


def environment_handles(marker_size: float = 5.5) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker=ENV_MARKERS[environment], linestyle="",
            markerfacecolor=ENV_COLORS[environment], markeredgecolor="none",
            label=environment, markersize=marker_size,
        )
        for environment in ("PC", "VR")
    ]


def subject_colors() -> dict[int, tuple]:
    cmap = plt.get_cmap("turbo", EXPECTED_SUBJECTS)
    return {subject: cmap(subject) for subject in range(EXPECTED_SUBJECTS)}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#D9D9D9", alpha=0.58, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_global_pca(
    frame: pd.DataFrame, explained: np.ndarray, analysis_dir: Path, dpi: int
) -> dict[str, str]:
    x_label, y_label = variance_labels(explained)
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    for environment in ("PC", "VR"):
        rows = frame[frame["environment"] == environment]
        ax.scatter(
            rows["PC1"], rows["PC2"], s=PCA_GLOBAL_SIZE,
            alpha=PCA_GLOBAL_ALPHA, color=ENV_COLORS[environment],
            marker=ENV_MARKERS[environment], linewidths=0, label=environment,
            rasterized=False,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(
        handles=environment_handles(), title="Environment", loc="upper right",
        frameon=True, framealpha=0.90, borderpad=0.45, handletextpad=0.45,
    )
    style_axis(ax)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.13, top=0.98)
    return save_figure(fig, analysis_dir / "pca_global_pc_vs_vr_paper", dpi)


def plot_subject_pca(
    frame: pd.DataFrame, explained: np.ndarray, analysis_dir: Path, dpi: int
) -> dict[str, str]:
    x_label, y_label = variance_labels(explained)
    fig, axes = plt.subplots(
        7, 3, figsize=(7.1, 12.8), sharex=True, sharey=True
    )
    for subject, ax in enumerate(axes.ravel()):
        subject_rows = frame[frame["subject_label"] == subject]
        if len(subject_rows) != EXPECTED_PER_SUBJECT:
            raise AssertionError(f"Subject {subject}: expected 240 PCA rows")
        for environment in ("PC", "VR"):
            rows = subject_rows[subject_rows["environment"] == environment]
            if len(rows) != EXPECTED_PER_SUBJECT_ENVIRONMENT:
                raise AssertionError(
                    f"Subject {subject}, {environment}: expected 120 PCA rows"
                )
            ax.scatter(
                rows["PC1"], rows["PC2"], s=PCA_SUBJECT_SIZE,
                alpha=PCA_SUBJECT_ALPHA, color=ENV_COLORS[environment],
                marker=ENV_MARKERS[environment], linewidths=0, zorder=2,
            )
            ax.scatter(
                rows["PC1"].mean(), rows["PC2"].mean(), s=PCA_CENTROID_SIZE,
                alpha=0.96, color=ENV_COLORS[environment], marker="X",
                edgecolors="white", linewidths=0.65, zorder=4,
            )
        ax.set_title(f"Subject {subject + 1}", fontweight="semibold", pad=3.5)
        style_axis(ax)

    legend_handles = environment_handles()
    legend_handles.extend(
        [
            Line2D(
                [0], [0], marker="X", linestyle="", color=ENV_COLORS[environment],
                markeredgecolor="white", markeredgewidth=0.65,
                label=f"{environment} centroid", markersize=6.5,
            )
            for environment in ("PC", "VR")
        ]
    )
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=4, frameon=False, columnspacing=1.15, handletextpad=0.40,
    )
    fig.supxlabel(x_label, y=0.025)
    fig.supylabel(y_label, x=0.018)
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.055, top=0.963, hspace=0.34, wspace=0.16
    )
    return save_figure(fig, analysis_dir / "pca_per_subject_pc_vs_vr_paper", dpi)


def padded_limits(values: np.ndarray, fraction: float = 0.035) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    padding = fraction * (maximum - minimum)
    return minimum - padding, maximum + padding


def plot_tsne(frame: pd.DataFrame, analysis_dir: Path, dpi: int) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.35), sharex=True, sharey=True)
    x_limits = padded_limits(frame["tSNE1"].to_numpy(dtype=float))
    y_limits = padded_limits(frame["tSNE2"].to_numpy(dtype=float))

    for environment in ("PC", "VR"):
        rows = frame[frame["environment"] == environment]
        axes[0].scatter(
            rows["tSNE1"], rows["tSNE2"], s=TSNE_SIZE, alpha=TSNE_ALPHA,
            color=ENV_COLORS[environment], marker=ENV_MARKERS[environment],
            linewidths=0, label=environment,
        )
    axes[0].set_title("(a) Environment", loc="left", fontweight="bold", pad=5)
    axes[0].legend(
        handles=environment_handles(5.0), loc="upper right", frameon=True,
        framealpha=0.90, borderpad=0.40, handletextpad=0.35,
    )

    colors = subject_colors()
    for subject in range(EXPECTED_SUBJECTS):
        for environment in ("PC", "VR"):
            rows = frame[
                (frame["subject_label"] == subject)
                & (frame["environment"] == environment)
            ]
            axes[1].scatter(
                rows["tSNE1"], rows["tSNE2"], s=TSNE_SIZE, alpha=TSNE_ALPHA,
                color=colors[subject], marker=ENV_MARKERS[environment],
                linewidths=0,
            )
    axes[1].set_title(
        "(b) Subject and Environment", loc="left", fontweight="bold", pad=5
    )
    environment_legend = axes[1].legend(
        handles=[
            Line2D(
                [0], [0], marker=ENV_MARKERS[environment], linestyle="",
                color="#4D4D4D", markerfacecolor="#4D4D4D",
                markeredgecolor="none", label=environment, markersize=5.0,
            )
            for environment in ("PC", "VR")
        ],
        title="Environment", loc="upper right", frameon=True, framealpha=0.90,
        borderpad=0.40, handletextpad=0.35,
    )
    axes[1].add_artist(environment_legend)

    subject_handles_by_id = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=colors[subject],
            markeredgecolor="none", label=str(subject + 1), markersize=4.5,
        )
        for subject in range(EXPECTED_SUBJECTS)
    ]
    # Matplotlib fills legend columns first. Reorder handles so the visible
    # rows read 1--7, 8--14, and 15--21 from left to right.
    subject_handles = [
        subject_handles_by_id[row * 7 + column]
        for column in range(7)
        for row in range(3)
    ]
    axes[1].legend(
        handles=subject_handles, title="Subject", loc="upper center",
        bbox_to_anchor=(0.5, -0.20), ncol=7, frameon=False,
        columnspacing=0.55, handletextpad=0.18, borderaxespad=0,
        labelspacing=0.35,
    )

    for ax in axes:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("t-SNE 1")
        style_axis(ax)
    axes[0].set_ylabel("t-SNE 2")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.94, wspace=0.16)
    return save_figure(fig, analysis_dir / "tsne_global_pc_vs_vr_paper", dpi)


def write_captions(analysis_dir: Path, explained: np.ndarray) -> Path:
    pc1 = 100.0 * explained[0]
    pc2 = 100.0 * explained[1]
    text = rf"""% Publication-ready appendix captions generated from saved analysis outputs.
\caption{{Global principal-component projection of 5,040 unique epochs represented by the frozen REVE encoder. A single PCA transformation was fitted globally to the 512-dimensional embeddings; PC1 and PC2 explain {pc1:.2f}\% and {pc2:.2f}\% of the variance, respectively. PC and VR observations show substantial overlap in this two-dimensional projection. This visualization is descriptive and does not constitute a test for the absence of an environment effect.}}

\caption{{Participant-level PCA views of frozen REVE embeddings. All 21 panels use the same global PCA transformation, with 120 PC and 120 VR embeddings shown for each participant. Crosses indicate participant- and environment-specific centroids in the shared PCA coordinate system, providing a participant-level view of environment-dependent representation structure.}}

\caption{{t-SNE visualization of 5,040 frozen REVE embeddings. The 512-dimensional representations were first projected onto the first 50 components of the global PCA and then embedded with t-SNE. The left panel distinguishes PC and VR environments, while the right panel encodes participant by color and environment by marker. This nonlinear projection is qualitative only; environment displacement is assessed quantitatively using the separate subject-level centroid-distance analysis.}}
"""
    path = analysis_dir / "reve_appendix_figure_captions.tex"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.dpi < 400:
        raise ValueError("--dpi must be at least 400")
    plt.rcParams.update(STYLE)

    pca, tsne, explained, validation = load_inputs(args.analysis_dir)
    outputs = {
        "global_pca": plot_global_pca(pca, explained, args.analysis_dir, args.dpi),
        "per_subject_pca": plot_subject_pca(pca, explained, args.analysis_dir, args.dpi),
        "tsne": plot_tsne(tsne, args.analysis_dir, args.dpi),
    }
    captions = write_captions(args.analysis_dir, explained)

    validation.update(
        {
            "outputs": outputs,
            "captions": str(captions.resolve()),
            "dpi": args.dpi,
            "figures": {
                "global_pca_inches": [7.1, 4.8],
                "per_subject_pca_inches": [7.1, 12.8],
                "tsne_inches": [7.1, 4.35],
            },
            "markers": {
                "PC": "blue circle",
                "VR": "orange triangle",
                "global_pca_size": PCA_GLOBAL_SIZE,
                "global_pca_alpha": PCA_GLOBAL_ALPHA,
                "per_subject_pca_size": PCA_SUBJECT_SIZE,
                "per_subject_pca_alpha": PCA_SUBJECT_ALPHA,
                "per_subject_centroid_size": PCA_CENTROID_SIZE,
                "tsne_size": TSNE_SIZE,
                "tsne_alpha": TSNE_ALPHA,
            },
            "plotting_parameters": {
                "font_family": STYLE["font.family"],
                "base_font_points": STYLE["font.size"],
                "axis_label_points": STYLE["axes.labelsize"],
                "panel_title_points": STYLE["axes.titlesize"],
                "tick_label_points": STYLE["xtick.labelsize"],
                "legend_points": STYLE["legend.fontsize"],
                "global_pca_legend": "inside upper right",
                "per_subject_pca_legend": "shared upper center, four columns",
                "tsne_environment_legends": "inside upper right of each panel",
                "tsne_subject_legend": "below right panel, seven columns by three rows",
            },
            "figure_validation": {
                "global_pca_points": EXPECTED_TOTAL,
                "per_subject_pca_panels": EXPECTED_SUBJECTS,
                "per_subject_pca_points_per_panel": EXPECTED_PER_SUBJECT,
                "per_subject_pca_pc_per_panel": EXPECTED_PER_SUBJECT_ENVIRONMENT,
                "per_subject_pca_vr_per_panel": EXPECTED_PER_SUBJECT_ENVIRONMENT,
                "per_subject_shared_global_coordinates": True,
                "tsne_points_per_panel": EXPECTED_TOTAL,
                "tsne_panels_share_coordinate_limits": True,
            },
            "final_set_excludes_pca_subject_environment": True,
            "embeddings_labels_statistics_modified": False,
        }
    )
    validation_path = args.analysis_dir / "reve_appendix_figures_paper_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()


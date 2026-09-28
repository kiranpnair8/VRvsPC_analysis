#!/usr/bin/env python3
"""Reconstruct unique outer-test REVE embeddings and analyze PC/VR structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


EXPECTED_SUBJECTS = 21
EXPECTED_EPOCHS_PER_SUBJECT_ENV = 120
EXPECTED_ENV_COUNT = EXPECTED_SUBJECTS * EXPECTED_EPOCHS_PER_SUBJECT_ENV
EXPECTED_TOTAL = EXPECTED_ENV_COUNT * 2
EXPECTED_DIM = 512
EXPECTED_GENUINE_PER_FOLD_ENV = EXPECTED_SUBJECTS * 24
ENV_SPECS = (("PC", "pc", "PC->PC"), ("VR", "vr", "VR->VR"))
ENV_COLORS = {"PC": "#3366CC", "VR": "#D9532F"}
ENV_MARKERS = {"PC": "o", "VR": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze unique frozen-REVE embeddings from supervised outer-test caches."
    )
    parser.add_argument(
        "--cv5_npz",
        default="./out/cv5/cv5_verification_dataset_lphp10_50.npz",
    )
    parser.add_argument("--cache_dir", default="./out/cv5/cache/reve")
    parser.add_argument("--out_dir", default="./reve_embedding_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--bootstrap_ci_repeats", type=int, default=10000)
    return parser.parse_args()


def load_meta(data: np.lib.npyio.NpzFile) -> dict:
    if "meta_json" not in data:
        raise KeyError("CV5 dataset is missing meta_json")
    value = data["meta_json"]
    if hasattr(value, "item"):
        value = value.item()
    return json.loads(str(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_single_cache(cache_dir: Path, scenario: str, fold: int) -> Path:
    patterns = [
        f"{scenario}_fold{fold}_test_*.npz",
        f"{scenario.replace('->', '_to_')}_fold{fold}_test_*.npz",
    ]
    matches = sorted({path for pattern in patterns for path in cache_dir.glob(pattern)})
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cache for {scenario} fold {fold} test; "
            f"found {len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def subject_epoch_lookup(y_epoch: np.ndarray) -> np.ndarray:
    lookup = np.full(y_epoch.shape[0], -1, dtype=np.int32)
    for subject in sorted(np.unique(y_epoch).astype(int)):
        indices = np.flatnonzero(y_epoch == subject)
        if indices.size != EXPECTED_EPOCHS_PER_SUBJECT_ENV:
            raise AssertionError(
                f"Subject {subject} has {indices.size} epochs; "
                f"expected {EXPECTED_EPOCHS_PER_SUBJECT_ENV}"
            )
        lookup[indices] = np.arange(indices.size, dtype=np.int32)
    if (lookup < 0).any():
        raise AssertionError("Failed to assign within-subject epoch indices")
    return lookup


def reconstruct_unique_embeddings(
    cv5_npz: Path, cache_dir: Path
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict, pd.DataFrame]:
    if not cv5_npz.is_file():
        raise FileNotFoundError(f"CV5 dataset not found: {cv5_npz}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"REVE supervised cache directory not found: {cache_dir}")

    data = np.load(cv5_npz, allow_pickle=True)
    meta = load_meta(data)
    inverse_subject_map = {
        int(label): str(subject_name)
        for subject_name, label in meta.get("subject_map", {}).items()
    }

    parts: dict[str, list[np.ndarray]] = {
        "Z": [],
        "subject_label": [],
        "environment": [],
        "environment_epoch": [],
        "subject_epoch": [],
        "outer_test_fold": [],
    }
    manifest_rows = []

    for env_name, env_key, scenario in ENV_SPECS:
        y_epoch = np.asarray(data[f"y_epoch_{env_key}"], dtype=np.int32)
        if y_epoch.shape != (EXPECTED_ENV_COUNT,):
            raise AssertionError(
                f"y_epoch_{env_key} has shape {y_epoch.shape}; expected {(EXPECTED_ENV_COUNT,)}"
            )
        local_epoch = subject_epoch_lookup(y_epoch)

        for fold in range(5):
            cache_path = find_single_cache(cache_dir, scenario, fold)
            with np.load(cache_path, allow_pickle=False) as cache:
                if set(cache.files) != {"Z"}:
                    raise AssertionError(
                        f"Unexpected keys in {cache_path}: {cache.files}; expected only Z"
                    )
                Z_trials = np.asarray(cache["Z"], dtype=np.float32)

            prefix = f"{env_key}_fold{fold}_test"
            y_trial = np.asarray(data[f"y_{prefix}"], dtype=np.int32)
            claimed_id = np.asarray(data[f"cid_{prefix}"], dtype=np.int32)
            source_epoch = np.asarray(
                data[f"trial_source_epoch_{prefix}"], dtype=np.int64
            )
            if not (Z_trials.shape[0] == y_trial.size == claimed_id.size == source_epoch.size):
                raise AssertionError(
                    f"Row mismatch for {scenario} fold {fold}: Z={Z_trials.shape[0]}, "
                    f"y={y_trial.size}, cid={claimed_id.size}, source={source_epoch.size}"
                )
            if Z_trials.ndim != 2 or Z_trials.shape[1] != EXPECTED_DIM:
                raise AssertionError(
                    f"Unexpected embedding shape in {cache_path}: {Z_trials.shape}"
                )

            genuine = y_trial == 1
            if int(genuine.sum()) != EXPECTED_GENUINE_PER_FOLD_ENV:
                raise AssertionError(
                    f"{scenario} fold {fold} has {int(genuine.sum())} genuine rows; "
                    f"expected {EXPECTED_GENUINE_PER_FOLD_ENV}"
                )
            source = source_epoch[genuine]
            if np.unique(source).size != source.size:
                raise AssertionError(f"Duplicate genuine source epochs in {scenario} fold {fold}")
            if source.min() < 0 or source.max() >= y_epoch.size:
                raise AssertionError(f"Out-of-range source epoch in {scenario} fold {fold}")

            true_subject = y_epoch[source]
            if not np.array_equal(claimed_id[genuine], true_subject):
                raise AssertionError(
                    f"Claimed and true subject mismatch among genuine rows for {scenario} fold {fold}"
                )

            count = source.size
            parts["Z"].append(Z_trials[genuine])
            parts["subject_label"].append(true_subject.astype(np.int32))
            parts["environment"].append(np.full(count, env_name, dtype="<U2"))
            parts["environment_epoch"].append(source.astype(np.int32))
            parts["subject_epoch"].append(local_epoch[source])
            parts["outer_test_fold"].append(np.full(count, fold, dtype=np.int32))
            manifest_rows.append(
                {
                    "environment": env_name,
                    "scenario": scenario,
                    "fold": fold,
                    "cache_path": str(cache_path.resolve()),
                    "cache_size_bytes": cache_path.stat().st_size,
                    "cache_sha256": sha256_file(cache_path),
                    "cache_rows": Z_trials.shape[0],
                    "genuine_rows_retained": count,
                    "embedding_dim": Z_trials.shape[1],
                }
            )

    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    assert arrays["Z"].shape == (EXPECTED_TOTAL, EXPECTED_DIM)
    for key in (
        "subject_label",
        "environment",
        "environment_epoch",
        "subject_epoch",
        "outer_test_fold",
    ):
        assert arrays[key].shape == (EXPECTED_TOTAL,), f"Bad shape for {key}: {arrays[key].shape}"

    subjects = np.unique(arrays["subject_label"])
    assert subjects.size == EXPECTED_SUBJECTS, f"Expected 21 subjects, found {subjects.size}"
    for environment in ("PC", "VR"):
        env_mask = arrays["environment"] == environment
        assert int(env_mask.sum()) == EXPECTED_ENV_COUNT
        pairs = np.column_stack(
            [arrays["subject_label"][env_mask], arrays["environment_epoch"][env_mask]]
        )
        assert np.unique(pairs, axis=0).shape[0] == EXPECTED_ENV_COUNT
        assert np.unique(arrays["environment_epoch"][env_mask]).size == EXPECTED_ENV_COUNT

    for subject in subjects:
        subject_mask = arrays["subject_label"] == subject
        assert int(subject_mask.sum()) == 240
        for environment in ("PC", "VR"):
            count = int((subject_mask & (arrays["environment"] == environment)).sum())
            assert count == 120, f"Subject {subject}, {environment}: expected 120, found {count}"

    physical_keys = np.array(
        [
            f"{environment}:{epoch}"
            for environment, epoch in zip(
                arrays["environment"], arrays["environment_epoch"]
            )
        ]
    )
    assert np.unique(physical_keys).size == EXPECTED_TOTAL
    assert np.isfinite(arrays["Z"]).all(), "Embeddings contain NaN or infinity"

    subject_names = np.array(
        [inverse_subject_map.get(int(label), f"subject_{int(label)}") for label in arrays["subject_label"]]
    )
    metadata = pd.DataFrame(
        {
            "row_index": np.arange(EXPECTED_TOTAL, dtype=np.int32),
            "subject_label": arrays["subject_label"],
            "subject_name": subject_names,
            "environment": arrays["environment"],
            "environment_epoch": arrays["environment_epoch"],
            "subject_epoch": arrays["subject_epoch"],
            "outer_test_fold": arrays["outer_test_fold"],
        }
    )
    return arrays, metadata, meta, pd.DataFrame(manifest_rows)


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def subject_colors(subjects: np.ndarray) -> dict[int, tuple]:
    cmap = plt.get_cmap("turbo", len(subjects))
    return {int(subject): cmap(i) for i, subject in enumerate(subjects)}


def plot_pca_environment(frame: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for environment in ("PC", "VR"):
        rows = frame[frame["environment"] == environment]
        ax.scatter(
            rows["PC1"], rows["PC2"], s=12, alpha=0.45,
            color=ENV_COLORS[environment], marker=ENV_MARKERS[environment],
            label=environment, linewidths=0,
        )
        ax.scatter(
            rows["PC1"].mean(), rows["PC2"].mean(), s=130,
            color=ENV_COLORS[environment], marker="X", edgecolor="black", linewidth=0.8,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Environment")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "pca_global_pc_vs_vr", dpi)


def plot_subject_environment(
    frame: pd.DataFrame, x_col: str, y_col: str, stem: Path, dpi: int
) -> None:
    subjects = np.sort(frame["subject_label"].unique())
    colors = subject_colors(subjects)
    fig, ax = plt.subplots(figsize=(12, 8))
    for subject in subjects:
        for environment in ("PC", "VR"):
            rows = frame[
                (frame["subject_label"] == subject) & (frame["environment"] == environment)
            ]
            ax.scatter(
                rows[x_col], rows[y_col], s=12, alpha=0.42,
                color=colors[int(subject)], marker=ENV_MARKERS[environment], linewidths=0,
            )
    subject_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[int(subject)],
               label=str(int(subject)), markersize=6)
        for subject in subjects
    ]
    env_handles = [
        Line2D([0], [0], marker=ENV_MARKERS[environment], linestyle="", color="#444444",
               label=environment, markersize=7)
        for environment in ("PC", "VR")
    ]
    fig.legend(
        handles=subject_handles, title="Subject label", bbox_to_anchor=(0.99, 0.98),
        loc="upper right", ncol=2, fontsize=8,
    )
    ax.legend(handles=env_handles, title="Environment", loc="lower right")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.2)
    fig.subplots_adjust(right=0.78, left=0.09, bottom=0.10, top=0.97)
    save_figure(fig, stem, dpi)


def plot_pca_per_subject(frame: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    subjects = np.sort(frame["subject_label"].unique())
    fig, axes = plt.subplots(7, 3, figsize=(12, 21), sharex=True, sharey=True)
    for ax, subject in zip(axes.ravel(), subjects):
        for environment in ("PC", "VR"):
            rows = frame[
                (frame["subject_label"] == subject) & (frame["environment"] == environment)
            ]
            ax.scatter(
                rows["PC1"], rows["PC2"], s=9, alpha=0.48,
                color=ENV_COLORS[environment], marker=ENV_MARKERS[environment],
                linewidths=0, label=environment,
            )
            ax.scatter(
                rows["PC1"].mean(), rows["PC2"].mean(), s=55,
                color=ENV_COLORS[environment], marker="X", edgecolor="black", linewidth=0.5,
            )
        ax.set_title(f"Subject {int(subject)}", fontsize=10)
        ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure(fig, out_dir / "pca_per_subject_pc_vs_vr", dpi)


def run_pca(
    Z: np.ndarray, metadata: pd.DataFrame, out_dir: Path, dpi: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pca = PCA(n_components=EXPECTED_DIM, svd_solver="full")
    coordinates = pca.fit_transform(Z).astype(np.float32)
    explained = pca.explained_variance_ratio_.astype(np.float64)
    frame = metadata.copy()
    frame["PC1"] = coordinates[:, 0]
    frame["PC2"] = coordinates[:, 1]
    np.savez_compressed(
        out_dir / "pca_coordinates.npz",
        coordinates=coordinates,
        explained_variance_ratio=explained,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
    )
    frame.to_csv(out_dir / "pca_coordinates_2d.csv", index=False)
    pd.DataFrame(
        {"component": np.arange(1, explained.size + 1), "explained_variance_ratio": explained}
    ).to_csv(out_dir / "pca_explained_variance.csv", index=False)
    plot_pca_environment(frame, out_dir, dpi)
    plot_subject_environment(frame, "PC1", "PC2", out_dir / "pca_subject_environment", dpi)
    plot_pca_per_subject(frame, out_dir, dpi)
    return coordinates, explained, frame


def run_tsne(
    pca_coordinates: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
    seed: int,
    perplexity: float,
    max_iter: int,
    dpi: int,
) -> tuple[np.ndarray, dict]:
    parameters = {
        "input": "first 50 PCs fitted on all 5040 frozen REVE embeddings",
        "n_components": 2,
        "perplexity": float(perplexity),
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "max_iter": int(max_iter),
        "init": "pca",
        "metric": "euclidean",
        "method": "barnes_hut",
        "angle": 0.5,
        "random_state": int(seed),
    }
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=max_iter,
        init="pca",
        metric="euclidean",
        method="barnes_hut",
        angle=0.5,
        random_state=seed,
    )
    coordinates = tsne.fit_transform(pca_coordinates[:, :50]).astype(np.float32)
    frame = metadata.copy()
    frame["tSNE1"] = coordinates[:, 0]
    frame["tSNE2"] = coordinates[:, 1]
    frame.to_csv(out_dir / "tsne_coordinates.csv", index=False)
    (out_dir / "tsne_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for environment in ("PC", "VR"):
        rows = frame[frame["environment"] == environment]
        axes[0].scatter(
            rows["tSNE1"], rows["tSNE2"], s=10, alpha=0.45,
            color=ENV_COLORS[environment], marker=ENV_MARKERS[environment],
            linewidths=0, label=environment,
        )
    axes[0].set_title("Environment")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    subjects = np.sort(frame["subject_label"].unique())
    colors = subject_colors(subjects)
    for subject in subjects:
        for environment in ("PC", "VR"):
            rows = frame[
                (frame["subject_label"] == subject) & (frame["environment"] == environment)
            ]
            axes[1].scatter(
                rows["tSNE1"], rows["tSNE2"], s=9, alpha=0.42,
                color=colors[int(subject)], marker=ENV_MARKERS[environment], linewidths=0,
            )
    subject_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[int(subject)],
               label=str(int(subject)), markersize=6)
        for subject in subjects
    ]
    fig.legend(
        handles=subject_handles, title="Subject label", bbox_to_anchor=(0.995, 0.98),
        loc="upper right", ncol=2, fontsize=8,
    )
    axes[1].set_title("Subject color; environment marker")
    axes[1].grid(alpha=0.2)
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    fig.tight_layout(rect=(0, 0, 0.87, 1))
    save_figure(fig, out_dir / "tsne_global_pc_vs_vr", dpi)
    return coordinates, parameters


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("Cannot compute cosine distance for a zero centroid")
    return float(1.0 - np.dot(a, b) / denominator)


def row_distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(a - b, axis=1)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(denominator == 0):
        raise ValueError("Zero bootstrap centroid encountered in cosine distance")
    return 1.0 - numerator / denominator


def bootstrap_within_reference(
    X: np.ndarray, metric: str, repeats: int, rng: np.random.Generator, chunk_size: int = 50
) -> tuple[float, float]:
    n = X.shape[0]
    distances = []
    for start in range(0, repeats, chunk_size):
        size = min(chunk_size, repeats - start)
        first = rng.integers(0, n, size=(size, n))
        second = rng.integers(0, n, size=(size, n))
        centroid_first = X[first].mean(axis=1)
        centroid_second = X[second].mean(axis=1)
        distances.append(row_distances(centroid_first, centroid_second, metric))
    values = np.concatenate(distances)
    return float(values.mean()), float(values.std(ddof=1))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_ci(
    values: np.ndarray, statistic, repeats: int, rng: np.random.Generator
) -> tuple[float, float]:
    samples = rng.integers(0, values.size, size=(repeats, values.size))
    estimates = np.apply_along_axis(statistic, 1, values[samples])
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_test(
    frame: pd.DataFrame,
    metric: str,
    ci_repeats: int,
    rng: np.random.Generator,
) -> dict:
    cross = frame[f"cross_{metric}"].to_numpy(dtype=float)
    within = frame[f"within_reference_{metric}"].to_numpy(dtype=float)
    difference = cross - within
    shapiro = stats.shapiro(difference)
    normal = bool(shapiro.pvalue >= 0.05)

    result = {
        "metric": metric,
        "n_subjects": difference.size,
        "cross_mean": float(cross.mean()),
        "cross_sd": float(cross.std(ddof=1)),
        "cross_median": float(np.median(cross)),
        "cross_q1": float(np.percentile(cross, 25)),
        "cross_q3": float(np.percentile(cross, 75)),
        "within_mean": float(within.mean()),
        "within_sd": float(within.std(ddof=1)),
        "within_median": float(np.median(within)),
        "within_q1": float(np.percentile(within, 25)),
        "within_q3": float(np.percentile(within, 75)),
        "difference_mean": float(difference.mean()),
        "difference_sd": float(difference.std(ddof=1)),
        "difference_median": float(np.median(difference)),
        "normality_test": "Shapiro-Wilk on paired subject differences",
        "normality_statistic": float(shapiro.statistic),
        "normality_p": float(shapiro.pvalue),
    }

    if normal:
        test = stats.ttest_rel(cross, within)
        standard_error = stats.sem(difference)
        ci = stats.t.interval(
            0.95, df=difference.size - 1, loc=difference.mean(), scale=standard_error
        )
        result.update(
            {
                "test": "paired t-test",
                "test_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "effect_size_name": "Cohen_dz",
                "effect_size": float(difference.mean() / difference.std(ddof=1)),
                "difference_ci_type": "95% t CI for mean paired difference",
                "difference_ci_low": float(ci[0]),
                "difference_ci_high": float(ci[1]),
                "descriptive_summary": "mean_sd",
            }
        )
    else:
        test = stats.wilcoxon(cross, within, alternative="two-sided", zero_method="wilcox")
        ci = bootstrap_ci(difference, np.median, ci_repeats, rng)
        result.update(
            {
                "test": "Wilcoxon signed-rank",
                "test_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "effect_size_name": "rank_biserial_correlation",
                "effect_size": float(rank_biserial(difference)),
                "difference_ci_type": "95% subject-bootstrap CI for median paired difference",
                "difference_ci_low": float(ci[0]),
                "difference_ci_high": float(ci[1]),
                "descriptive_summary": "median_iqr",
            }
        )
    return result


def plot_distances(frame: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, ("euclidean", "cosine")):
        within = frame[f"within_reference_{metric}"].to_numpy()
        cross = frame[f"cross_{metric}"].to_numpy()
        for first, second in zip(within, cross):
            ax.plot([0, 1], [first, second], color="#999999", alpha=0.55, linewidth=0.8)
        ax.scatter(np.zeros_like(within), within, color="#777777", s=28, zorder=3)
        ax.scatter(np.ones_like(cross), cross, color="#A23B72", s=28, zorder=3)
        ax.set_xticks([0, 1], ["Within-env\nreference", "PC-VR\ncentroids"])
        ax.set_ylabel(f"{metric.capitalize()} distance")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "subject_centroid_distance_comparison", dpi)


def run_distance_analysis(
    Z: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
    seed: int,
    bootstrap_repeats: int,
    ci_repeats: int,
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    centroids = []
    subjects = np.sort(metadata["subject_label"].unique())
    for subject in subjects:
        subject_rows = metadata["subject_label"].to_numpy() == subject
        pc = Z[subject_rows & (metadata["environment"].to_numpy() == "PC")]
        vr = Z[subject_rows & (metadata["environment"].to_numpy() == "VR")]
        if pc.shape != (120, EXPECTED_DIM) or vr.shape != (120, EXPECTED_DIM):
            raise AssertionError(f"Unexpected subject arrays for {subject}: PC={pc.shape}, VR={vr.shape}")
        mu_pc = pc.mean(axis=0)
        mu_vr = vr.mean(axis=0)
        centroids.extend([(int(subject), "PC", mu_pc), (int(subject), "VR", mu_vr)])
        row = {
            "subject_label": int(subject),
            "cross_euclidean": float(np.linalg.norm(mu_pc - mu_vr)),
            "cross_cosine": cosine_distance(mu_pc, mu_vr),
        }
        for metric in ("euclidean", "cosine"):
            pc_mean, pc_sd = bootstrap_within_reference(
                pc, metric, bootstrap_repeats, rng
            )
            vr_mean, vr_sd = bootstrap_within_reference(
                vr, metric, bootstrap_repeats, rng
            )
            row[f"within_pc_{metric}"] = pc_mean
            row[f"within_pc_bootstrap_sd_{metric}"] = pc_sd
            row[f"within_vr_{metric}"] = vr_mean
            row[f"within_vr_bootstrap_sd_{metric}"] = vr_sd
            row[f"within_reference_{metric}"] = (pc_mean + vr_mean) / 2.0
            row[f"difference_{metric}"] = row[f"cross_{metric}"] - row[f"within_reference_{metric}"]
        rows.append(row)

    distance_frame = pd.DataFrame(rows)
    distance_frame.to_csv(out_dir / "subject_level_distances.csv", index=False)
    np.savez_compressed(
        out_dir / "subject_environment_centroids.npz",
        centroids=np.stack([item[2] for item in centroids]).astype(np.float32),
        subject_label=np.array([item[0] for item in centroids], dtype=np.int32),
        environment=np.array([item[1] for item in centroids]),
    )

    tests = [
        paired_test(distance_frame, metric, ci_repeats, rng)
        for metric in ("euclidean", "cosine")
    ]
    adjusted = holm_adjust([item["p_value"] for item in tests])
    for item, adjusted_p in zip(tests, adjusted):
        item["multiple_comparison_method"] = "Holm correction across Euclidean and cosine tests"
        item["p_value_holm"] = adjusted_p
        item["significant_holm_0.05"] = bool(adjusted_p < 0.05)
    test_frame = pd.DataFrame(tests)
    test_frame.to_csv(out_dir / "statistical_test_results.csv", index=False)
    plot_distances(distance_frame, out_dir, dpi)
    return distance_frame, test_frame


def print_validation(arrays: dict[str, np.ndarray], manifest: pd.DataFrame) -> None:
    print("\nEmbedding reconstruction validation")
    print(f"  cache files used: {len(manifest)}")
    print(f"  Z shape: {arrays['Z'].shape}")
    print(f"  subjects: {np.unique(arrays['subject_label']).size}")
    for environment in ("PC", "VR"):
        mask = arrays["environment"] == environment
        print(f"  {environment} unique epochs: {int(mask.sum())}")
    counts = pd.crosstab(arrays["subject_label"], arrays["environment"])
    print("  per-subject environment counts all equal 120:", bool((counts == 120).all().all()))
    print("  total per subject all equal 240:", bool((counts.sum(axis=1) == 240).all()))
    print("  duplicate physical epochs: 0")


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300")
    if args.tsne_max_iter < 250:
        raise ValueError("--tsne_max_iter must be at least 250")
    if args.bootstrap_repeats < 100:
        raise ValueError("--bootstrap_repeats must be at least 100")

    cv5_npz = Path(args.cv5_npz)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata, source_meta, manifest = reconstruct_unique_embeddings(
        cv5_npz, cache_dir
    )
    print_validation(arrays, manifest)
    np.savez_compressed(out_dir / "reve_unique_embeddings.npz", **arrays)
    metadata.to_csv(out_dir / "embedding_metadata.csv", index=False)
    manifest.to_csv(out_dir / "cache_manifest.csv", index=False)

    pca_coordinates, explained, _ = run_pca(
        arrays["Z"], metadata, out_dir, args.dpi
    )
    print(
        f"\nPCA explained variance: PC1={explained[0]:.8f} "
        f"({explained[0] * 100:.4f}%), PC2={explained[1]:.8f} "
        f"({explained[1] * 100:.4f}%)"
    )
    _, tsne_parameters = run_tsne(
        pca_coordinates,
        metadata,
        out_dir,
        args.seed,
        args.tsne_perplexity,
        args.tsne_max_iter,
        args.dpi,
    )
    _, tests = run_distance_analysis(
        arrays["Z"],
        metadata,
        out_dir,
        args.seed,
        args.bootstrap_repeats,
        args.bootstrap_ci_repeats,
        args.dpi,
    )

    analysis_meta = {
        "cv5_npz": str(cv5_npz.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "seed": args.seed,
        "embedding_shape": list(arrays["Z"].shape),
        "pca_pc1_explained_variance_ratio": float(explained[0]),
        "pca_pc2_explained_variance_ratio": float(explained[1]),
        "tsne": tsne_parameters,
        "within_environment_reference": (
            "For each subject/environment, two independent bootstrap samples of 120 epochs "
            "were averaged into centroids; their distance was computed and averaged across "
            f"{args.bootstrap_repeats} repetitions. PC and VR reference distances were averaged."
        ),
        "statistical_unit": "subject (n=21)",
        "multiple_comparison_correction": "Holm across Euclidean and cosine paired tests",
        "source_meta": source_meta,
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(analysis_meta, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)

    print("\nSubject-level statistical results")
    columns = [
        "metric", "n_subjects", "test", "test_statistic", "p_value",
        "p_value_holm", "effect_size_name", "effect_size",
        "difference_ci_low", "difference_ci_high",
    ]
    print(tests[columns].to_string(index=False))
    for row in tests.to_dict("records"):
        direction = "larger" if row["difference_mean"] > 0 else "smaller"
        evidence = "detectable" if row["significant_holm_0.05"] else "not statistically detectable"
        print(
            f"  {row['metric']}: PC-VR displacement was {direction} than the within-environment "
            f"reference; the subject-level difference was {evidence} after Holm correction."
        )
    print(
        "\nInterpretation boundary: these results test environment-dependent structure in frozen "
        "REVE representations. They do not establish that REVE causes authentication degradation "
        "and do not isolate raw-EEG or classifier-level effects."
    )
    print(f"\nSaved analysis outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


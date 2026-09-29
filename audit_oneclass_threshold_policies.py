#!/usr/bin/env python3
"""Audit one-class CV5 score semantics and compare validation thresholds.

The script consumes existing REVE embedding caches. It does not run REVE,
modify the CV5 dataset, tune on test data, or overwrite existing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from cv5_metrics import (
    compute_eer,
    metrics_at_threshold,
    threshold_at_validation_eer,
    threshold_for_far_zero,
)


SCENARIOS = {
    "PC->PC": {"train": "pc", "val": "pc", "test": "pc"},
    "VR->VR": {"train": "vr", "val": "vr", "test": "vr"},
    "PC->VR": {"train": "pc", "val": "pc", "test": "vr"},
    "VR->PC": {"train": "vr", "val": "vr", "test": "pc"},
    "Mixed->PC": {"train": "mixed", "val": "mixed", "test": "pc"},
    "Mixed->VR": {"train": "mixed", "val": "mixed", "test": "vr"},
}


POLICIES = ("near_zero_far", "far_le_1pct", "far_le_5pct", "validation_eer")
POLICY_LABELS = {
    "near_zero_far": "Current near-zero FAR",
    "far_le_1pct": "FAR <= 1%",
    "far_le_5pct": "FAR <= 5%",
    "validation_eer": "Validation EER",
}
POLICY_COLORS = {
    "near_zero_far": "#222222",
    "far_le_1pct": "#7B3294",
    "far_le_5pct": "#008837",
    "validation_eer": "#E66101",
}
POLICY_LINESTYLES = {
    "near_zero_far": "-",
    "far_le_1pct": "--",
    "far_le_5pct": "-.",
    "validation_eer": ":",
}


def subject_data(oneclass, env: str, fold: int, subject: int):
    env_data = oneclass[env]
    fold_data = env_data[fold] if fold in env_data else env_data[str(fold)]
    return fold_data[subject] if subject in fold_data else fold_data[str(subject)]


def concat_sets(sets, key: str) -> np.ndarray:
    return np.concatenate([np.asarray(item[key]) for item in sets], axis=0)


def oneclass_splits(oneclass, scenario: str, fold: int, subject: int):
    """Mirror the corrected runner's split selection without importing REVE code."""
    spec = SCENARIOS[scenario]
    test_set = subject_data(oneclass, spec["test"], fold, subject)
    if spec["train"] == "mixed":
        source_sets = [
            subject_data(oneclass, "pc", fold, subject),
            subject_data(oneclass, "vr", fold, subject),
        ]
        Xtr = concat_sets(source_sets, "Xtr")
        Xva = concat_sets(source_sets, "Xva")
        yva = concat_sets(source_sets, "yva")
    else:
        source_set = subject_data(oneclass, spec["train"], fold, subject)
        Xtr = np.asarray(source_set["Xtr"])
        Xva = np.asarray(source_set["Xva"])
        yva = np.asarray(source_set["yva"])
    return (
        Xtr,
        Xva,
        yva.astype(int),
        np.asarray(test_set["Xte"]),
        np.asarray(test_set["yte"]).astype(int),
    )


def fit_model(model_name: str, Ztr: np.ndarray, args):
    """Mirror the corrected runner's unchanged one-class estimators."""
    if model_name == "OCSVM":
        model = OneClassSVM(kernel=args.kernel, nu=args.nu, gamma=args.gamma)
    elif model_name == "OCkNN":
        model = NearestNeighbors(n_neighbors=args.n_neighbors, metric=args.metric)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.fit(Ztr)
    return model


def score_model(model_name: str, model, Z: np.ndarray) -> np.ndarray:
    """Return scores in the audited larger-is-more-genuine orientation."""
    if model_name == "OCSVM":
        return model.decision_function(Z).ravel()
    if model_name == "OCkNN":
        distances, _ = model.kneighbors(Z)
        return -distances.mean(axis=1)
    raise ValueError(f"Unknown model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CV5 one-class thresholds using cached REVE embeddings."
    )
    parser.add_argument("--npz", default="./out/cv5/cv5_verification_dataset_lphp10_50.npz")
    parser.add_argument("--cache_dir", default="./out/cv5/cache/reve_oneclass")
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--out_dir", default="./out/cv5/oneclass_threshold_audit")
    parser.add_argument("--model", choices=["OCSVM", "OCkNN", "all"], default="all")
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def array_digest(X: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(X.shape).encode("utf-8"))
    digest.update(str(X.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(X).view(np.uint8))
    return digest.hexdigest()


def expected_cache_path(
    cache_dir: Path, model_dir: Path, ch_names: list[str], X: np.ndarray, tag: str
) -> Path:
    payload = {
        "tag": tag,
        "model_dir": str(model_dir.resolve()),
        "ch_names": list(ch_names),
        "x_digest": array_digest(X),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{tag}_{key}.npz"


def load_cached_Z(
    cache_dir: Path,
    model_dir: Path,
    ch_names: list[str],
    X: np.ndarray,
    tag: str,
) -> tuple[np.ndarray, Path]:
    path = expected_cache_path(cache_dir, model_dir, ch_names, X, tag)
    if not path.is_file():
        matches = sorted(cache_dir.glob(f"{tag}_*.npz"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one existing cache for {tag}; exact={path}, matches={matches}"
            )
        path = matches[0]
    with np.load(path, allow_pickle=False) as archive:
        if "Z" not in archive.files:
            raise ValueError(f"Cache lacks Z: {path}")
        Z = archive["Z"]
    if Z.ndim != 2 or Z.shape[0] != X.shape[0]:
        raise AssertionError(f"Cache shape mismatch for {tag}: Z={Z.shape}, X={X.shape}")
    if not np.isfinite(Z).all():
        raise AssertionError(f"Non-finite embedding in {path}")
    return Z, path


def threshold_for_far_constraint(scores, labels, target_far: float) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    if not 0.0 <= target_far <= 1.0:
        raise ValueError(f"target_far must be in [0,1], found {target_far}")
    if scores.shape != labels.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Threshold selection requires equal-length scores and 0/1 labels")
    impostor = scores[labels == 0]
    genuine = scores[labels == 1]
    if impostor.size == 0 or genuine.size == 0:
        raise ValueError("Threshold selection requires genuine and impostor validation scores")

    max_false_accepts = int(np.floor(target_far * impostor.size + 1e-12))
    if max_false_accepts >= impostor.size:
        return float(-np.inf)
    descending = np.sort(impostor)[::-1]
    boundary = descending[max_false_accepts]
    threshold = float(np.nextafter(boundary, np.inf))

    metrics = metrics_at_threshold(scores, labels, threshold)
    if metrics["FAR"] > target_far + 1e-12:
        raise AssertionError(
            f"FAR-constrained threshold failed: target={target_far}, actual={metrics['FAR']}"
        )
    lower = float(np.nextafter(threshold, -np.inf))
    lower_far = metrics_at_threshold(scores, labels, lower)["FAR"]
    if lower_far <= target_far + 1e-12:
        raise AssertionError(
            "Selected threshold is not the lowest representable threshold satisfying FAR constraint"
        )
    return threshold


def select_thresholds(val_scores: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    thresholds = {
        "near_zero_far": threshold_for_far_zero(val_scores, y_val),
        "far_le_1pct": threshold_for_far_constraint(val_scores, y_val, 0.01),
        "far_le_5pct": threshold_for_far_constraint(val_scores, y_val, 0.05),
        "validation_eer": threshold_at_validation_eer(val_scores, y_val),
    }
    if thresholds["near_zero_far"] != threshold_for_far_constraint(val_scores, y_val, 0.0):
        raise AssertionError("Current near-zero-FAR helper differs from exact FAR<=0 construction")
    return thresholds


def distribution_summary(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    q05, q25, median, q75, q95 = np.percentile(scores, [5, 25, 50, 75, 95])
    return {
        "n": int(scores.size),
        "min": float(scores.min()),
        "p05": float(q05),
        "p25": float(q25),
        "median": float(median),
        "p75": float(q75),
        "p95": float(q95),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
    }


def assert_metric_definition(scores, labels, threshold, metrics) -> None:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    accepted = scores >= threshold
    impostor = labels == 0
    genuine = labels == 1
    expected_far = float(np.sum(accepted & impostor) / np.sum(impostor))
    expected_frr = float(np.sum((~accepted) & genuine) / np.sum(genuine))
    if not np.isclose(metrics["FAR"], expected_far, rtol=0, atol=1e-15):
        raise AssertionError("FAR is not false accepts divided by impostor trials")
    if not np.isclose(metrics["FRR"], expected_frr, rtol=0, atol=1e-15):
        raise AssertionError("FRR is not false rejects divided by genuine trials")


def diagnostic_plot(
    scores: pd.DataFrame, thresholds: pd.DataFrame, out_dir: Path, dpi: int
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    score_min = float(scores["score"].min())
    score_max = float(scores["score"].max())
    bins = np.linspace(score_min, score_max, 70)

    for ax, split, title in zip(axes, ("validation", "test"), ("(a) Validation", "(b) Test")):
        split_scores = scores[scores["split"] == split]
        for label, color, name in ((1, "#3366CC", "Genuine"), (0, "#D9532F", "Impostor")):
            values = split_scores.loc[split_scores["label"] == label, "score"].to_numpy()
            ax.hist(
                values, bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=0.28, linewidth=0, label=name,
            )
            ax.hist(
                values, bins=bins, density=True, histtype="step",
                color=color, alpha=0.9, linewidth=0.8,
            )

        for policy in POLICIES:
            values = thresholds.loc[thresholds["threshold_policy"] == policy, "threshold"]
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            ax.axvspan(q1, q3, color=POLICY_COLORS[policy], alpha=0.055, linewidth=0)
            ax.axvline(
                median, color=POLICY_COLORS[policy],
                linestyle=POLICY_LINESTYLES[policy], linewidth=1.25,
                label=f"{POLICY_LABELS[policy]} threshold, median (IQR)"
                if split == "validation" else None,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("OC-kNN score (negative mean neighbor distance)")
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.91, wspace=0.13)
    stem = out_dir / "ocknn_pc_to_pc_fold0_score_distributions"
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fold_and_summary_tables(subject_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "validation_FAR", "validation_FRR", "test_FAR", "test_FRR",
        "test_balanced_accuracy", "test_EER",
    ]
    fold_df = (
        subject_df.groupby(
            ["model", "scenario", "fold", "threshold_policy", "seed"], as_index=False
        )
        .agg(
            threshold_mean=("threshold", "mean"),
            threshold_sd=("threshold", lambda values: values.std(ddof=1)),
            validation_FAR=("validation_FAR", "mean"),
            validation_FRR=("validation_FRR", "mean"),
            test_FAR=("test_FAR", "mean"),
            test_FRR=("test_FRR", "mean"),
            test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            test_EER=("test_EER", "mean"),
            n_subjects=("subject", "nunique"),
        )
    )
    policy_order = {policy: index for index, policy in enumerate(POLICIES)}
    fold_df["_policy_order"] = fold_df["threshold_policy"].map(policy_order)
    fold_df = fold_df.sort_values(
        ["model", "scenario", "_policy_order", "fold"]
    ).drop(columns="_policy_order")
    if not (fold_df["n_subjects"] == 21).all():
        raise AssertionError("Every fold/policy must macro-average exactly 21 subjects")

    rows = []
    for key, group in fold_df.groupby(["model", "scenario", "threshold_policy", "seed"]):
        if sorted(group["fold"].tolist()) != list(range(5)):
            raise AssertionError(f"Expected exactly folds 0..4 for {key}")
        model, scenario, policy, seed = key
        row = {
            "model": model,
            "scenario": scenario,
            "threshold_policy": policy,
            "seed": seed,
            "n_folds": 5,
        }
        output_names = {
            "test_FAR": "far",
            "test_FRR": "frr",
            "test_balanced_accuracy": "balanced_accuracy",
            "test_EER": "test_eer",
        }
        for source, output in output_names.items():
            row[f"{output}_mean"] = float(group[source].mean())
            row[f"{output}_sd"] = float(group[source].std(ddof=1))
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df["_policy_order"] = summary_df["threshold_policy"].map(policy_order)
    summary_df = summary_df.sort_values(
        ["model", "scenario", "_policy_order"]
    ).drop(columns="_policy_order")
    return fold_df, summary_df


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    npz_path = Path(args.npz)
    cache_dir = Path(args.cache_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)

    dataset = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(dataset["meta_json"]))
    oneclass = dataset["oneclass"].item()
    subjects = sorted(int(subject) for subject in meta["subject_map"].values())
    folds = list(range(int(meta["n_folds"])))
    if len(subjects) != 21 or folds != list(range(5)):
        raise AssertionError(f"Expected 21 subjects and folds 0..4; got {subjects}, {folds}")
    models = ["OCSVM", "OCkNN"] if args.model == "all" else [args.model]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    ch_names = list(meta["ch_names"])

    subject_rows = []
    cache_rows = []
    diagnostic_score_rows = []
    diagnostic_threshold_rows = []
    observed_labels = set()

    for model_name in models:
        for scenario in scenarios:
            for fold in folds:
                for subject in subjects:
                    Xtr, Xva, yva, Xte, yte = oneclass_splits(
                        oneclass, scenario, fold, subject
                    )
                    observed_labels.update(np.unique(yva).astype(int).tolist())
                    observed_labels.update(np.unique(yte).astype(int).tolist())
                    if set(np.unique(yva)) != {0, 1} or set(np.unique(yte)) != {0, 1}:
                        raise AssertionError("One-class validation/test labels must be exactly {0,1}")
                    expected_val_genuine = 36 if SCENARIOS[scenario]["val"] == "mixed" else 18
                    expected_val_impostor = 720 if SCENARIOS[scenario]["val"] == "mixed" else 360
                    counts = {
                        "val_genuine": int(np.sum(yva == 1)),
                        "val_impostor": int(np.sum(yva == 0)),
                        "test_genuine": int(np.sum(yte == 1)),
                        "test_impostor": int(np.sum(yte == 0)),
                    }
                    expected_counts = {
                        "val_genuine": expected_val_genuine,
                        "val_impostor": expected_val_impostor,
                        "test_genuine": 24,
                        "test_impostor": 480,
                    }
                    if counts != expected_counts:
                        raise AssertionError(
                            f"Unexpected trial counts for {scenario}, fold {fold}, "
                            f"subject {subject}: {counts} != {expected_counts}"
                        )
                    common = f"{model_name}_{scenario}_fold{fold}_subj{subject}".replace(
                        "->", "_to_"
                    )
                    arrays = {"train": Xtr, "val": Xva, "test": Xte}
                    embeddings = {}
                    for split, X in arrays.items():
                        Z, path = load_cached_Z(
                            cache_dir, model_dir, ch_names, X, f"{common}_{split}"
                        )
                        embeddings[split] = Z
                        cache_rows.append(
                            {
                                "model": model_name,
                                "scenario": scenario,
                                "fold": fold,
                                "subject": subject,
                                "split": split,
                                "path": str(path.resolve()),
                                "rows": int(Z.shape[0]),
                                "dimensions": int(Z.shape[1]),
                            }
                        )

                    clf = fit_model(model_name, embeddings["train"], args)
                    val_scores = score_model(model_name, clf, embeddings["val"])
                    test_scores = score_model(model_name, clf, embeddings["test"])
                    if not np.isfinite(val_scores).all() or not np.isfinite(test_scores).all():
                        raise AssertionError("Non-finite one-class score")
                    thresholds = select_thresholds(val_scores, yva)
                    independent_test_eer = compute_eer(test_scores, yte)

                    for policy, threshold in thresholds.items():
                        val_metrics = metrics_at_threshold(val_scores, yva, threshold)
                        test_metrics = metrics_at_threshold(test_scores, yte, threshold)
                        assert_metric_definition(val_scores, yva, threshold, val_metrics)
                        assert_metric_definition(test_scores, yte, threshold, test_metrics)
                        subject_rows.append(
                            {
                                "model": model_name,
                                "scenario": scenario,
                                "fold": fold,
                                "subject": subject,
                                "threshold_policy": policy,
                                "threshold": threshold,
                                "validation_FAR": val_metrics["FAR"],
                                "validation_FRR": val_metrics["FRR"],
                                "test_FAR": test_metrics["FAR"],
                                "test_FRR": test_metrics["FRR"],
                                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                                "test_EER": independent_test_eer,
                                "seed": args.seed,
                            }
                        )

                    if model_name == "OCkNN" and scenario == "PC->PC" and fold == 0:
                        for split, split_scores, labels in (
                            ("validation", val_scores, yva), ("test", test_scores, yte)
                        ):
                            diagnostic_score_rows.extend(
                                {
                                    "model": model_name,
                                    "scenario": scenario,
                                    "fold": fold,
                                    "subject": subject,
                                    "split": split,
                                    "label": int(label),
                                    "score": float(score),
                                }
                                for score, label in zip(split_scores, labels)
                            )
                        diagnostic_threshold_rows.extend(
                            {
                                "subject": subject,
                                "threshold_policy": policy,
                                "threshold": threshold,
                            }
                            for policy, threshold in thresholds.items()
                        )
                print(f"Audited {model_name} {scenario} fold {fold}")

    if observed_labels != {0, 1}:
        raise AssertionError(f"Expected only labels 0/1; observed {observed_labels}")

    subject_df = pd.DataFrame(subject_rows)
    expected_subject_rows = len(models) * len(scenarios) * 5 * 21 * len(POLICIES)
    if len(subject_df) != expected_subject_rows:
        raise AssertionError(f"Expected {expected_subject_rows} subject rows, found {len(subject_df)}")
    fold_df, summary_df = fold_and_summary_tables(subject_df)

    subject_path = out_dir / "oneclass_threshold_policy_subject_results.csv"
    fold_path = out_dir / "oneclass_threshold_policy_per_fold.csv"
    summary_path = out_dir / "oneclass_cv5_threshold_policy_comparison.csv"
    cache_path = out_dir / "oneclass_threshold_policy_cache_manifest.csv"
    subject_df.to_csv(subject_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(cache_rows).to_csv(cache_path, index=False)

    if diagnostic_score_rows:
        diagnostic_scores = pd.DataFrame(diagnostic_score_rows)
        diagnostic_thresholds = pd.DataFrame(diagnostic_threshold_rows)
        diagnostic_scores.to_csv(out_dir / "ocknn_pc_to_pc_fold0_scores.csv", index=False)
        diagnostic_thresholds.to_csv(
            out_dir / "ocknn_pc_to_pc_fold0_subject_thresholds.csv", index=False
        )

        summary_rows = []
        for (split, label), group in diagnostic_scores.groupby(["split", "label"]):
            row = {
                "split": split,
                "class": "genuine" if label == 1 else "impostor",
                "label": int(label),
            }
            row.update(distribution_summary(group["score"].to_numpy()))
            summary_rows.append(row)
        score_summary = pd.DataFrame(summary_rows)
        score_summary.to_csv(out_dir / "ocknn_pc_to_pc_fold0_score_summary.csv", index=False)

        diagnostic_fold = fold_df[
            (fold_df["model"] == "OCkNN")
            & (fold_df["scenario"] == "PC->PC")
            & (fold_df["fold"] == 0)
        ].copy()
        diagnostic_fold.insert(
            4, "policy_label", diagnostic_fold["threshold_policy"].map(POLICY_LABELS)
        )
        diagnostic_fold.to_csv(
            out_dir / "ocknn_pc_to_pc_fold0_threshold_table.csv", index=False
        )
        diagnostic_plot(diagnostic_scores, diagnostic_thresholds, out_dir, args.dpi)

        diagnostic_counts = {
            "per_subject": {
                "n_validation_genuine": 18,
                "n_validation_impostor": 360,
                "n_test_genuine": 24,
                "n_test_impostor": 480,
            },
            "pooled_across_21_subject_models": {
                "n_validation_genuine": int(
                    np.sum(
                        (diagnostic_scores["split"] == "validation")
                        & (diagnostic_scores["label"] == 1)
                    )
                ),
                "n_validation_impostor": int(
                    np.sum(
                        (diagnostic_scores["split"] == "validation")
                        & (diagnostic_scores["label"] == 0)
                    )
                ),
                "n_test_genuine": int(
                    np.sum(
                        (diagnostic_scores["split"] == "test")
                        & (diagnostic_scores["label"] == 1)
                    )
                ),
                "n_test_impostor": int(
                    np.sum(
                        (diagnostic_scores["split"] == "test")
                        & (diagnostic_scores["label"] == 0)
                    )
                ),
            },
        }
        (out_dir / "ocknn_pc_to_pc_fold0_counts.json").write_text(
            json.dumps(diagnostic_counts, indent=2), encoding="utf-8"
        )

        print("\nOC-kNN PC->PC fold 0 pooled score distributions:")
        print(json.dumps(diagnostic_counts, indent=2))
        print(score_summary.to_string(index=False))
        print("\nOC-kNN PC->PC fold 0 macro-averaged threshold comparison:")
        print(
            diagnostic_fold[
                [
                    "policy_label", "validation_FAR", "validation_FRR",
                    "test_FAR", "test_FRR", "test_balanced_accuracy", "test_EER",
                ]
            ].to_string(index=False)
        )

    audit = {
        "dataset": str(npz_path.resolve()),
        "dataset_size_bytes": npz_path.stat().st_size,
        "existing_results_modified": False,
        "reve_rerun": False,
        "models_refit_from_cached_embeddings": True,
        "labels": {"genuine": 1, "impostor": 0, "negative_one_present": False},
        "score_orientation": {
            "OCSVM": "decision_function; larger means more inlier-like/genuine",
            "OCkNN": "negative mean k-neighbor distance; larger means closer/more genuine",
            "acceptance_rule": "score >= validation-selected threshold",
        },
        "threshold_selection": {
            "source": "validation scores and labels only",
            "test_used": False,
            "near_zero_far": "nextafter(max validation impostor score, +infinity)",
            "far_constraints": "lowest representable threshold satisfying FAR <= target",
            "validation_eer": "validation threshold minimizing absolute FAR-FRR difference",
            "test_eer": "computed independently from complete test scores; never an operating threshold",
        },
        "metric_definitions": {
            "FAR": "accepted impostor trials / all impostor trials",
            "FRR": "rejected genuine trials / all genuine trials",
        },
        "aggregation": "subjects macro-averaged within fold; five folds summarized with ddof=1",
        "models": models,
        "scenarios": scenarios,
        "folds": folds,
        "subjects": subjects,
        "policies": list(POLICIES),
        "subject_result_rows": len(subject_df),
        "fold_result_rows": len(fold_df),
        "summary_rows": len(summary_df),
    }
    (out_dir / "oneclass_threshold_policy_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"\nSaved audit outputs to {out_dir}")


if __name__ == "__main__":
    main()


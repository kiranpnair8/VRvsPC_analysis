#!/usr/bin/env python3
"""CPU-only frozen-REVE geometry with training-only CV5 enrollment centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, roc_curve

from analyze_reve_embeddings import (
    holm_adjust,
    rank_biserial,
    reconstruct_unique_embeddings,
    sha256_file,
)


SCENARIOS = {
    "PC->PC": (("PC",), "PC"),
    "VR->VR": (("VR",), "VR"),
    "PC->VR": (("PC",), "VR"),
    "VR->PC": (("VR",), "PC"),
    "Mixed->PC": (("PC", "VR"), "PC"),
    "Mixed->VR": (("PC", "VR"), "VR"),
}
TARGET_SCENARIOS = {
    "PC": ("PC->PC", "VR->PC", "Mixed->PC"),
    "VR": ("VR->VR", "PC->VR", "Mixed->VR"),
}
CONDITIONS = ("Matched", "Cross", "Mixed")
COLORS = ("#3366CC", "#D9532F", "#218C74")
PREFIX = "reve_embedding"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv5_npz", type=Path, default=Path("out/cv5/cv5_verification_dataset_lphp10_50.npz"))
    parser.add_argument("--embeddings_npz", type=Path, default=None,
                        help="Unique embeddings NPZ; otherwise discover it or use existing outer-test caches")
    parser.add_argument("--cache_dir", type=Path, default=Path("out/cv5/cache/reve"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/reve_geometry_cv5"))
    parser.add_argument("--fig_dir", type=Path, default=Path("figures/reve_geometry_cv5"))
    parser.add_argument("--distance", choices=("euclidean", "cosine", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--hist_bins", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def load_json(archive, key):
    return json.loads(str(archive[key].item()))


def split_indices(folds, env, subject, fold, split):
    return np.asarray(folds[env.lower()][str(subject)][str(fold)][split], dtype=np.int64)


def load_inputs(args):
    with np.load(args.cv5_npz, allow_pickle=False) as archive:
        meta = load_json(archive, "meta_json")
        folds = load_json(archive, "folds_json")
        labels = {env: archive[f"y_epoch_{env.lower()}"] for env in ("PC", "VR")}
        # Compare against the actual saved authentication test trial construction.
        genuine_tests = {}
        for env in ("PC", "VR"):
            for fold in range(5):
                prefix = f"{env.lower()}_fold{fold}_test"
                y = archive[f"y_{prefix}"]
                source = archive[f"trial_source_epoch_{prefix}"]
                cid = archive[f"cid_{prefix}"]
                if y.shape != source.shape or y.shape != cid.shape or not np.isin(y, [0, 1]).all():
                    raise AssertionError(f"Invalid saved test metadata: {prefix}")
                genuine = y == 1
                if not np.array_equal(cid[genuine], labels[env][source[genuine]]):
                    raise AssertionError(f"Genuine identity mismatch: {prefix}")
                genuine_tests[(env, fold)] = source[genuine]

    embedding_path = args.embeddings_npz
    if embedding_path is None:
        candidate = Path("reve_embedding_analysis/reve_unique_embeddings.npz")
        embedding_path = candidate if candidate.is_file() else None
    manifest = pd.DataFrame()
    if embedding_path is not None:
        with np.load(embedding_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in (
                "Z", "subject_label", "environment", "environment_epoch",
                "subject_epoch", "outer_test_fold",
            )}
        source_info = {"mode": "saved_unique_embeddings", "path": str(embedding_path.resolve()),
                       "sha256": sha256_file(embedding_path)}
    else:
        arrays, _, _, manifest = reconstruct_unique_embeddings(args.cv5_npz, args.cache_dir)
        source_info = {"mode": "existing_supervised_outer_test_caches",
                       "cache_dir": str(args.cache_dir.resolve())}
    subjects, bank = validate_inputs(arrays, labels, folds, meta, genuine_tests)
    return subjects, bank, folds, meta, manifest, source_info


def validate_inputs(arrays, labels, folds, meta, genuine_tests):
    if arrays["Z"].shape != (5040, 512) or not np.isfinite(arrays["Z"]).all():
        raise AssertionError("Expected finite Z with shape (5040,512)")
    for key in ("subject_label", "environment", "environment_epoch", "subject_epoch", "outer_test_fold"):
        if arrays[key].shape != (5040,):
            raise AssertionError(f"Bad metadata shape: {key}")
    subjects = np.sort(np.unique(arrays["subject_label"]).astype(int))
    if len(subjects) != 21 or sorted(int(x) for x in meta["subject_map"].values()) != subjects.tolist():
        raise AssertionError("Expected the same 21 participants as the CV5 subject_map")
    if meta["n_folds"] != 5 or set(np.unique(arrays["environment"])) != {"PC", "VR"}:
        raise AssertionError("Expected five folds and PC/VR environments")
    bank = {}
    for env in ("PC", "VR"):
        mask = arrays["environment"] == env
        epoch = arrays["environment_epoch"][mask].astype(int)
        if len(epoch) != 2520 or not np.array_equal(np.sort(epoch), np.arange(2520)):
            raise AssertionError(f"Duplicate/missing physical epoch in {env}")
        order = np.argsort(epoch)
        Z = arrays["Z"][mask][order].astype(np.float64)
        true = arrays["subject_label"][mask][order]
        outer = arrays["outer_test_fold"][mask][order]
        local = arrays["subject_epoch"][mask][order]
        if not np.array_equal(true, labels[env]):
            raise AssertionError(f"Embedding subjects differ from y_epoch_{env.lower()}")
        for subject in subjects:
            owned = np.flatnonzero(true == subject)
            if len(owned) != 120 or not np.array_equal(local[owned], np.arange(120)):
                raise AssertionError("Incorrect within-subject epoch mapping")
            all_test = []
            for fold in range(5):
                tr, va, te = [split_indices(folds, env, subject, fold, name) for name in ("train", "val", "test")]
                if tuple(map(len, (tr, va, te))) != (78, 18, 24):
                    raise AssertionError("Incorrect original split sizes")
                union = np.concatenate([tr, va, te])
                if not np.array_equal(np.sort(union), owned):
                    raise AssertionError("Duplicate, missing, wrong-subject or overlapping split epoch")
                if not np.all(outer[te] == fold):
                    raise AssertionError("Embedding outer_test_fold differs from saved folds_json")
                all_test.extend(te)
            if not np.array_equal(np.sort(all_test), owned):
                raise AssertionError("Epochs must be tested exactly once")
        for fold in range(5):
            expected = np.concatenate([split_indices(folds, env, s, fold, "test") for s in subjects])
            if not np.array_equal(np.sort(genuine_tests[(env, fold)]), np.sort(expected)):
                raise AssertionError("Saved genuine test trials differ from the outer test fold")
        bank[env] = {"Z": Z, "subject": true, "local_epoch": local}
    return subjects, bank


def distance_matrices(Z, centroids, metrics):
    result = {}
    if "euclidean" in metrics:
        result["euclidean"] = cdist(Z, centroids, metric="euclidean")
    if "cosine" in metrics:
        # Explicit convention: zero/zero distance 0; zero/nonzero distance 1.
        zn = np.linalg.norm(Z, axis=1)
        cn = np.linalg.norm(centroids, axis=1)
        unit_z = np.divide(Z, zn[:, None], out=np.zeros_like(Z), where=zn[:, None] > 0)
        unit_c = np.divide(centroids, cn[:, None], out=np.zeros_like(centroids), where=cn[:, None] > 0)
        distances = 1.0 - unit_z @ unit_c.T
        distances[np.ix_(zn == 0, cn == 0)] = 0.0
        if np.any(distances < -1e-10) or np.any(distances > 2 + 1e-10):
            raise AssertionError("Cosine distance outside [0,2]")
        result["cosine"] = np.clip(distances, 0, 2)
    for metric, values in result.items():
        if not np.isfinite(values).all() or np.any(values < 0):
            raise AssertionError(f"Non-finite or negative {metric} distance")
    return result


def separation(genuine, impostor, edges):
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    vg, vi = genuine.var(ddof=1), impostor.var(ddof=1)
    denominator = np.sqrt(0.5 * (vg + vi))
    margin = float(impostor.mean() - genuine.mean())
    dprime = float(margin / denominator) if denominator > 0 else np.nan
    scores = -np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    far, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    # Match cv5_metrics.compute_eer's unique-score threshold/tie convention,
    # without its quadratic scan. Exclude the additional ROC infinity threshold.
    valid = np.flatnonzero(np.isfinite(thresholds))[::-1]
    index = valid[np.argmin(np.abs(far[valid] - (1 - tpr[valid])))]
    p = np.histogram(genuine, bins=edges)[0] / len(genuine)
    q = np.histogram(impostor, bins=edges)[0] / len(impostor)
    return {
        "n_genuine": len(genuine), "n_impostor": len(impostor),
        "mean_genuine_distance": float(genuine.mean()),
        "std_genuine_distance": float(genuine.std(ddof=1)),
        "median_genuine_distance": float(np.median(genuine)),
        "mean_impostor_distance": float(impostor.mean()),
        "std_impostor_distance": float(impostor.std(ddof=1)),
        "median_impostor_distance": float(np.median(impostor)),
        "margin": margin, "d_prime": dprime,
        "cohen_style_separation": dprime,
        "histogram_overlap_coefficient": float(np.minimum(p, q).sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "embedding_space_eer": float((far[index] + 1 - tpr[index]) / 2),
    }


def construct_distances(subjects, bank, folds, metrics):
    tables, enrollment_rows, centroids_saved, audits = [], [], [], []
    for fold in range(5):
        tests = {env: np.concatenate([split_indices(folds, env, s, fold, "test") for s in subjects])
                 for env in ("PC", "VR")}
        for scenario, (sources, target) in SCENARIOS.items():
            centroids = []
            for subject in subjects:
                pieces = []
                for env in sources:
                    indices = split_indices(folds, env, subject, fold, "train")
                    held_out = np.concatenate([split_indices(folds, env, s, fold, split)
                                               for s in subjects for split in ("val", "test")])
                    if np.intersect1d(indices, held_out).size:
                        raise AssertionError("Enrollment includes validation/test epoch")
                    pieces.append(bank[env]["Z"][indices])
                    enrollment_rows.extend({"fold": fold, "scenario": scenario,
                                            "subject": int(subject), "environment": env,
                                            "epoch_index": int(i)} for i in indices)
                centroid = np.concatenate(pieces).mean(axis=0)
                centroids.append(centroid)
                centroids_saved.append((fold, scenario, int(subject), centroid))
            centroids = np.stack(centroids)
            epoch = tests[target]
            true = bank[target]["subject"][epoch]
            same = true[:, None] == subjects[None, :]
            if not np.all(same.sum(axis=1) == 1) or not np.all((~same).sum(axis=1) == 20):
                raise AssertionError("Each test epoch needs one genuine and 20 impostor comparisons")
            matrices = distance_matrices(bank[target]["Z"][epoch], centroids, metrics)
            frame = pd.DataFrame({
                "fold": fold, "scenario": scenario, "test_environment": target,
                "enrollment_environment": "+".join(sources),
                "true_subject": np.repeat(true, 21),
                "claimed_subject": np.tile(subjects, len(epoch)),
                "trial_type": np.where(same.ravel(), "genuine", "impostor"),
                "test_epoch_index": np.repeat(epoch, 21),
                "subject_epoch": np.repeat(bank[target]["local_epoch"][epoch], 21),
            })
            for metric, matrix in matrices.items():
                frame[f"{metric}_distance"] = matrix.ravel()
            tables.append(frame)
            audits.append({"fold": fold, "scenario": scenario, "test_environment": target,
                           "enrollment_environments": "+".join(sources),
                           "train_epochs_per_subject": 78 * len(sources),
                           "test_epochs": len(epoch), "genuine_count": int(same.sum()),
                           "impostor_count": int((~same).sum()),
                           "same_saved_test_fold": True, "training_only_enrollment": True,
                           "zero_test_norms": int(np.sum(np.linalg.norm(bank[target]["Z"][epoch], axis=1) == 0)),
                           "zero_centroid_norms": int(np.sum(np.linalg.norm(centroids, axis=1) == 0))})
            print(f"Distances: fold {fold} {scenario}, 504 genuine + 10080 impostor")
    observations = pd.concat(tables, ignore_index=True)
    if len(observations) != 317520:
        raise AssertionError("Unexpected total distance-pair count")
    return observations, pd.DataFrame(enrollment_rows), centroids_saved, pd.DataFrame(audits)


def summarize(observations, metrics, bins):
    edges = {metric: np.histogram_bin_edges(observations[f"{metric}_distance"], bins=bins)
             for metric in metrics}
    fold_rows, subject_rows = [], []
    for (fold, scenario), frame in observations.groupby(["fold", "scenario"], sort=True):
        for metric in metrics:
            column = f"{metric}_distance"
            base = {"fold": fold, "scenario": scenario, "distance_metric": metric,
                    "test_environment": SCENARIOS[scenario][1]}
            g = frame.loc[frame.trial_type == "genuine", column].to_numpy()
            i = frame.loc[frame.trial_type == "impostor", column].to_numpy()
            if (len(g), len(i)) != (504, 10080):
                raise AssertionError("Wrong fold/scenario observation counts")
            fold_rows.append({**base, **separation(g, i, edges[metric])})
            # Participant = true identity of the probe; average its 24 probes
            # and their 20 nonmatching centroid comparisons, not claimed ID.
            for subject, probes in frame.groupby("true_subject"):
                g = probes.loc[probes.trial_type == "genuine", column].to_numpy()
                i = probes.loc[probes.trial_type == "impostor", column].to_numpy()
                if (len(g), len(i)) != (24, 480):
                    raise AssertionError("Wrong participant observation counts")
                subject_rows.append({**base, "participant": subject, **separation(g, i, edges[metric])})
    fold_frame, participant_frame = pd.DataFrame(fold_rows), pd.DataFrame(subject_rows)
    columns = [c for c in fold_frame.columns if c not in (
        "fold", "scenario", "distance_metric", "test_environment", "n_genuine", "n_impostor")]
    means = participant_frame.groupby(["scenario", "distance_metric", "test_environment", "participant"], as_index=False)[columns].mean()
    counts = participant_frame.groupby(["scenario", "distance_metric", "participant"]).fold.nunique()
    if not (counts == 5).all():
        raise AssertionError("Every participant needs five folds")
    means["n_folds"] = 5
    rows = []
    for (scenario, metric, target), frame in fold_frame.groupby(["scenario", "distance_metric", "test_environment"]):
        row = {"scenario": scenario, "distance_metric": metric, "test_environment": target, "n_folds": 5}
        for column in columns:
            row[f"{column}_mean"] = frame[column].mean()
            row[f"{column}_sd"] = frame[column].std(ddof=1)
        rows.append(row)
    return fold_frame, participant_frame, means, pd.DataFrame(rows), edges


def paired_statistics(participants, metrics, seed):
    rows = []
    for metric in metrics:
        for outcome in ("mean_genuine_distance", "margin"):
            for target, scenarios in TARGET_SCENARIOS.items():
                subset = participants[(participants.distance_metric == metric) & (participants.test_environment == target)]
                pivot = subset.pivot(index="participant", columns="scenario", values=outcome)
                if pivot.shape != (21, 3) or not np.isfinite(pivot.to_numpy()).all():
                    raise AssertionError("Statistics require finite paired values from all 21 participants")
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    first, second = scenarios[a], scenarios[b]
                    x, y = pivot[first].to_numpy(), pivot[second].to_numpy()
                    diff = x - y
                    constant = np.ptp(diff) <= 1e-14
                    normality = stats.shapiro(diff) if not constant else None
                    normal = normality is not None and normality.pvalue >= 0.05
                    if np.all(diff == 0):
                        name, statistic, p, effect_name, effect = "identical pairs", 0., 1., "rank_biserial", 0.
                    elif normal:
                        test = stats.ttest_rel(x, y)
                        name, statistic, p = "paired t-test", test.statistic, test.pvalue
                        effect_name, effect = "Cohen_dz", diff.mean() / diff.std(ddof=1)
                    else:
                        test = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
                        name, statistic, p = "Wilcoxon signed-rank", test.statistic, test.pvalue
                        effect_name, effect = "rank_biserial", rank_biserial(diff)
                    rows.append({"distance_metric": metric, "outcome": outcome, "test_environment": target,
                                 "comparison": f"{CONDITIONS[a]} - {CONDITIONS[b]}",
                                 "first_scenario": first, "second_scenario": second, "n_participants": 21,
                                 "first_mean": x.mean(), "second_mean": y.mean(),
                                 "mean_paired_difference": diff.mean(), "median_paired_difference": np.median(diff),
                                 "shapiro_statistic": normality.statistic if normality else np.nan,
                                 "shapiro_p": normality.pvalue if normality else np.nan,
                                 "test": name, "test_statistic": statistic, "raw_p": p,
                                 "effect_size_name": effect_name, "effect_size": effect,
                                 "difference_direction": "first minus second", "seed": seed})
    frame = pd.DataFrame(rows)
    # Six planned comparisons per distance/outcome family: three per target.
    for (metric, outcome), group in frame.groupby(["distance_metric", "outcome"]):
        frame.loc[group.index, "holm_p"] = holm_adjust(group.raw_p.tolist())
        frame.loc[group.index, "holm_family"] = f"{metric}/{outcome}: 3 comparisons x 2 targets"
    return frame


def figure_stems(metrics):
    return [f"reve_genuine_impostor_{metric}_cv5" for metric in metrics] + [
        "reve_embedding_dprime_cv5", "reve_participant_genuine_distance_shift_cv5", "reve_participant_margin_cv5"]


def save_figure(fig, path, dpi):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def plot_all(observations, participants, metrics, edges, args):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42})
    # Pool outer-held-out probes for visualization only. Pair dependencies are
    # retained in the data; all inferential tests use participant averages.
    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.7), sharex=True, sharey=True)
        for row, (target, scenarios) in enumerate(TARGET_SCENARIOS.items()):
            for col, scenario in enumerate(scenarios):
                ax = axes[row, col]
                subset = observations[observations.scenario == scenario]
                for kind, color in (("genuine", COLORS[0]), ("impostor", COLORS[1])):
                    values = subset.loc[subset.trial_type == kind, f"{metric}_distance"]
                    ax.hist(values, bins=edges[metric], density=True, histtype="step",
                            color=color, linewidth=1.1, label=kind.capitalize())
                ax.set_title(f"({chr(97 + row * 3 + col)}) {scenario}", loc="left")
                style_axis(ax)
                if col == 0:
                    ax.set_ylabel(f"{target} test\nDensity")
                if row == 1:
                    ax.set_xlabel(f"{metric.capitalize()} distance")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.subplots_adjust(left=.09, right=.99, top=.90, bottom=.11, wspace=.17, hspace=.30)
        save_figure(fig, args.fig_dir / f"reve_genuine_impostor_{metric}_cv5", args.dpi)

    fig, axes = plt.subplots(1, len(metrics), figsize=(7.1 if len(metrics) == 2 else 3.5, 3.1), squeeze=False)
    rng = np.random.default_rng(args.seed)
    ci_rows = []
    for col, metric in enumerate(metrics):
        ax = axes[0, col]
        for j, target in enumerate(("PC", "VR")):
            for k, scenario in enumerate(TARGET_SCENARIOS[target]):
                values = participants.loc[(participants.scenario == scenario) & (participants.distance_metric == metric), "d_prime"].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Undefined participant d-prime (zero variance); cannot plot CI")
                boot = rng.choice(values, size=(args.bootstrap_repeats, 21), replace=True).mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                x = k + (-.13 if j == 0 else .13)
                # Draw interval endpoints directly; percentile CI need not contain the point estimate.
                ax.vlines(x, lo, hi, color=COLORS[j], linewidth=1.1)
                ax.scatter(x, values.mean(), color=COLORS[j], marker=("o", "s")[j], s=22,
                           label=f"{target} test" if k == 0 else None, zorder=3)
                ci_rows.append({"distance_metric": metric, "scenario": scenario, "n_participants": 21,
                                "mean_participant_d_prime": values.mean(), "ci95_low": lo, "ci95_high": hi,
                                "bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed})
        ax.set_xticks(range(3), CONDITIONS)
        ax.set_ylabel("Participant d-prime")
        ax.set_title(f"({chr(97 + col)}) {metric.capitalize()}", loc="left")
        style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .995))
    fig.subplots_adjust(left=.10, right=.98, bottom=.14, top=.80, wspace=.30)
    save_figure(fig, args.fig_dir / "reve_embedding_dprime_cv5", args.dpi)
    for outcome, stem, ylabel in (
        ("mean_genuine_distance", "reve_participant_genuine_distance_shift_cv5", "Mean genuine distance"),
        ("margin", "reve_participant_margin_cv5", "Impostor - genuine margin"),
    ):
        fig, axes = plt.subplots(len(metrics), 2, figsize=(7.1, 2.65 * len(metrics)), squeeze=False, sharey="row")
        for row, metric in enumerate(metrics):
            for col, target in enumerate(("PC", "VR")):
                ax = axes[row, col]
                subset = participants[(participants.distance_metric == metric) & (participants.test_environment == target)]
                pivot = subset.pivot(index="participant", columns="scenario", values=outcome)[list(TARGET_SCENARIOS[target])]
                for values in pivot.to_numpy():
                    ax.plot(range(3), values, color="#999999", linewidth=.65, alpha=.45)
                for k, color in enumerate(COLORS):
                    ax.scatter(np.full(21, k), pivot.iloc[:, k], s=14, color=color, zorder=3)
                ax.set_xticks(range(3), CONDITIONS)
                ax.set_title(f"({chr(97 + row * 2 + col)}) {target} test / {metric.capitalize()}", loc="left")
                if col == 0:
                    ax.set_ylabel(ylabel)
                style_axis(ax)
        fig.subplots_adjust(left=.11, right=.99, bottom=.10, top=.94, hspace=.35, wspace=.17)
        save_figure(fig, args.fig_dir / stem, args.dpi)
    return pd.DataFrame(ci_rows)


def guard_outputs(args, metrics):
    names = ["trial_distances", "separation_summary", "participant_summary", "participant_fold_summary",
             "separation_per_fold", "statistical_tests", "enrollment_indices", "sanity_checks",
             "cache_manifest", "dprime_ci"]
    paths = [args.out_dir / f"{PREFIX}_{name}_cv5.csv" for name in names]
    paths += [args.out_dir / "reve_geometry_cv5_manifest.json", args.out_dir / "reve_geometry_enrollment_centroids.npz"]
    if not args.skip_plots:
        paths += [args.fig_dir / f"{stem}.{suffix}" for stem in figure_stems(metrics) for suffix in ("pdf", "png")]
    collision = [str(path) for path in paths if path.exists()]
    if collision:
        raise FileExistsError(f"Existing outputs would be overwritten; choose a new --out_dir/--fig_dir: {collision}")


def main():
    args = parse_args()
    if args.hist_bins < 2 or args.bootstrap_repeats < 100 or args.dpi < 300:
        raise ValueError("Require hist_bins >=2, bootstrap_repeats >=100, dpi >=300")
    metrics = ["euclidean", "cosine"] if args.distance == "both" else [args.distance]
    guard_outputs(args, metrics)
    subjects, bank, folds, meta, cache_manifest, source_info = load_inputs(args)
    observations, enrollment, centroids, audit = construct_distances(subjects, bank, folds, metrics)
    fold_frame, subject_folds, participants, summary, edges = summarize(observations, metrics, args.hist_bins)
    tests = paired_statistics(participants, metrics, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"trial_distances": observations, "separation_summary": summary,
              "participant_summary": participants, "participant_fold_summary": subject_folds,
              "separation_per_fold": fold_frame, "statistical_tests": tests,
              "enrollment_indices": enrollment, "sanity_checks": audit}
    if not cache_manifest.empty:
        tables["cache_manifest"] = cache_manifest
    if not args.skip_plots:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        tables["dprime_ci"] = plot_all(observations, participants, metrics, edges, args)
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{PREFIX}_{name}_cv5.csv", index=False)
    np.savez_compressed(args.out_dir / "reve_geometry_enrollment_centroids.npz",
                        centroids=np.stack([c[3] for c in centroids]),
                        fold=np.array([c[0] for c in centroids]),
                        scenario=np.array([c[1] for c in centroids]),
                        subject=np.array([c[2] for c in centroids]))
    manifest = {"cv5_npz": str(args.cv5_npz.resolve()), "cv5_npz_sha256": sha256_file(args.cv5_npz),
                "embedding_source": source_info, "embeddings_recomputed": False,
                "centroid_source": "fold-specific training indices only; original unscaled 512-D REVE",
                "subject_map": meta["subject_map"], "scenarios": list(SCENARIOS),
                "participants": 21, "unique_embeddings": 5040, "embedding_dimension": 512,
                "distance_pair_rows": len(observations), "impostor_centroids_per_probe": 20,
                "epoch_index_convention": "environment-local global CV5 index; raw recording chronology unavailable",
                "statistical_unit": "participant; arithmetic average of five fold values before paired testing",
                "holm_families": "6 tests per distance/outcome: 3 enrollment comparisons x 2 fixed targets",
                "d_prime": "(mean impostor - mean genuine) / sqrt((sample variance genuine + sample variance impostor)/2)",
                "cohen_style_separation": "same equal-weight variance denominator as d-prime; not sample-size-pooled Cohen d",
                "embedding_space_eer": "analytical negative-distance EER, not authentication model EER",
                "eer_threshold_rule": "closest FAR/FRR at unique scores; lowest-score tie break; no operating threshold reported",
                "cosine_zero_norm_convention": "zero/zero=0; zero/nonzero=1; counts saved in sanity_checks",
                "histogram_overlap": "sum min bin probabilities; shared global equal-width edges, no KDE",
                "histogram_edges": {m: v.tolist() for m, v in edges.items()},
                "seed": args.seed, "bootstrap_repeats": args.bootstrap_repeats,
                "bootstrap_scope": "21 participant fold-averaged d-primes; percentile mean CI for figures",
                "metrics": metrics, "result_files": {name: str((args.out_dir / f'{PREFIX}_{name}_cv5.csv').resolve()) for name in tables}}
    (args.out_dir / "reve_geometry_cv5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nAnalytical embedding geometry: fold means (not verifier metrics)")
    display = ["scenario", "distance_metric", "mean_genuine_distance_mean", "mean_impostor_distance_mean",
               "d_prime_mean", "roc_auc_mean", "embedding_space_eer_mean"]
    print(summary[display].to_string(index=False))
    print("\nParticipant paired comparisons; n=21, first minus second")
    print(tests[["distance_metric", "outcome", "test_environment", "comparison", "mean_paired_difference",
                 "test", "raw_p", "holm_p", "effect_size"]].to_string(index=False))
    print(f"\nSaved results: {args.out_dir.resolve()}")
    if not args.skip_plots:
        print(f"Saved figures: {args.fig_dir.resolve()}")


if __name__ == "__main__":
    main()


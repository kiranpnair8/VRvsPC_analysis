#!/usr/bin/env python3
"""Run the sample-count-controlled mixed-environment CV5 ablation.

Only MixedMatched models are trained here. Existing Cross, MixedFull, and
Within fold results are imported unchanged from the corrected CV5 results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cv5_metrics import compute_eer, metrics_at_threshold, threshold_at_validation_eer
from cv5_split_utils import supervised_trials, trial_seed


MODELS = ("REVE+MLP", "EEGNet")
TARGETS = ("PC", "VR")
METRICS = ("accuracy", "balanced_accuracy", "far", "frr", "eer")
SELECTION_ENV_OFFSET = {"pc": 10_000, "vr": 20_000}
SELECTION_SPLIT_OFFSET = {"train": 100, "val": 200}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MixedMatched CV5 and combine it with saved supervised baselines."
    )
    parser.add_argument(
        "--npz", default="./out/cv5/cv5_verification_dataset_lphp10_50.npz"
    )
    parser.add_argument(
        "--existing_results", default="./out/cv5/results/supervised_cv5_per_fold.csv"
    )
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--cache_dir", default="./out/cv5/cache/reve_mixed_size_control")
    parser.add_argument("--out_dir", default="./out/cv5/mixed_size_control")
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reve_bs", type=int, default=64)
    parser.add_argument("--head_bs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Generate and validate deterministic subsets without training models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard resumable MixedMatched fold rows and retrain them.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_arrays(data, env: str, fold: int, split: str):
    prefix = f"{env}_fold{fold}_{split}"
    return (
        data[f"X_{prefix}"].astype(np.float32),
        data[f"y_{prefix}"].astype(np.int32),
        data[f"cid_{prefix}"].astype(np.int64),
    )


def selection_seed(base_seed: int, env: str, fold: int, split: str, subject: int) -> int:
    return int(
        base_seed
        + 800_000
        + SELECTION_ENV_OFFSET[env]
        + SELECTION_SPLIT_OFFSET[split]
        + fold * 1_000
        + subject
    )


def fold_subject_split(folds: dict, env: str, subject: int, fold: int, split: str) -> np.ndarray:
    subject_map = folds[env][str(subject)] if str(subject) in folds[env] else folds[env][subject]
    fold_map = subject_map[str(fold)] if str(fold) in subject_map else subject_map[fold]
    return np.asarray(fold_map[split], dtype=np.int64)


def select_matched_indices(
    folds: dict, subjects: list[int], fold: int, seed: int
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    for env in ("pc", "vr"):
        for split, count in (("train", 39), ("val", 9)):
            rows = []
            for subject in subjects:
                available = fold_subject_split(folds, env, subject, fold, split)
                expected = 78 if split == "train" else 18
                if available.size != expected:
                    raise AssertionError(
                        f"{env} subject {subject} fold {fold} {split}: "
                        f"expected {expected}, found {available.size}"
                    )
                rng = np.random.default_rng(
                    selection_seed(seed, env, fold, split, subject)
                )
                choice = np.sort(rng.choice(available, size=count, replace=False))
                if np.unique(choice).size != count:
                    raise AssertionError("Matched subset contains duplicate epochs")
                rows.append(choice)
            selected[f"{split}_{env}"] = np.stack(rows)
    return selected


def validate_and_save_splits(
    data,
    folds: dict,
    subjects: list[int],
    fold: int,
    selected: dict[str, np.ndarray],
    out_dir: Path,
    seed: int,
) -> list[dict]:
    repeated = select_matched_indices(folds, subjects, fold, seed)
    for key in selected:
        if not np.array_equal(selected[key], repeated[key]):
            raise AssertionError(f"Deterministic rerun mismatch for fold {fold}, {key}")

    sanity_rows = []
    test_rows = {"pc": [], "vr": []}
    for env in ("pc", "vr"):
        for subject_index, subject in enumerate(subjects):
            train = selected[f"train_{env}"][subject_index]
            val = selected[f"val_{env}"][subject_index]
            test = fold_subject_split(folds, env, subject, fold, "test")
            original_train = fold_subject_split(folds, env, subject, fold, "train")
            original_val = fold_subject_split(folds, env, subject, fold, "val")
            if not set(train).issubset(set(original_train)):
                raise AssertionError("Selected training epoch is outside original training partition")
            if not set(val).issubset(set(original_val)):
                raise AssertionError("Selected validation epoch is outside original validation partition")
            if any(np.unique(values).size != values.size for values in (train, val, test)):
                raise AssertionError("Matched split contains duplicate epoch indices")
            if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
                raise AssertionError("Train/validation/test leakage in matched subset")
            if train.size != 39 or val.size != 9 or test.size != 24:
                raise AssertionError("Incorrect matched subset size")
            test_rows[env].append(np.sort(test))
            sanity_rows.append(
                {
                    "fold": fold,
                    "subject": subject,
                    "environment": env.upper(),
                    "train_selected": int(train.size),
                    "validation_selected": int(val.size),
                    "test_held_out": int(test.size),
                    "train_seed": selection_seed(seed, env, fold, "train", subject),
                    "validation_seed": selection_seed(seed, env, fold, "val", subject),
                    "no_duplicates": True,
                    "no_overlap": True,
                    "same_original_test_fold": True,
                    "deterministic_rerun": True,
                }
            )

        y_test = data[f"y_{env}_fold{fold}_test"].astype(int)
        source_test = data[f"trial_source_epoch_{env}_fold{fold}_test"].astype(int)
        expected_test_union = np.concatenate(test_rows[env])
        if not set(source_test).issubset(set(expected_test_union)):
            raise AssertionError(f"Existing {env} test trials include a non-test epoch")
        genuine_sources = source_test[y_test == 1]
        if not np.array_equal(np.sort(genuine_sources), np.sort(expected_test_union)):
            raise AssertionError(f"Existing {env} genuine test trials differ from held-out fold")

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_dir / f"fold_{fold}_mixedmatched_indices.npz",
        subjects=np.asarray(subjects, dtype=np.int32),
        train_pc_indices=selected["train_pc"],
        train_vr_indices=selected["train_vr"],
        validation_pc_indices=selected["val_pc"],
        validation_vr_indices=selected["val_vr"],
        test_pc_indices=np.stack(test_rows["pc"]),
        test_vr_indices=np.stack(test_rows["vr"]),
        train_pc_seeds=np.asarray(
            [selection_seed(seed, "pc", fold, "train", subject) for subject in subjects]
        ),
        train_vr_seeds=np.asarray(
            [selection_seed(seed, "vr", fold, "train", subject) for subject in subjects]
        ),
        validation_pc_seeds=np.asarray(
            [selection_seed(seed, "pc", fold, "val", subject) for subject in subjects]
        ),
        validation_vr_seeds=np.asarray(
            [selection_seed(seed, "vr", fold, "val", subject) for subject in subjects]
        ),
    )
    return sanity_rows


def build_trials_for_fold(data, subjects: list[int], fold: int, selected: dict, seed: int):
    trial_sets = {}
    provenance = {}
    for split in ("train", "val"):
        for env in ("pc", "vr"):
            selected_matrix = selected[f"{split}_{env}"]
            selected_flat = selected_matrix.reshape(-1)
            X_epoch = data[f"X_epoch_{env}"]
            y_epoch = data[f"y_epoch_{env}"].astype(int)
            if not np.array_equal(
                y_epoch[selected_flat],
                np.repeat(subjects, selected_matrix.shape[1]),
            ):
                raise AssertionError(f"Selected {env} {split} epochs do not match subject labels")
            trials = supervised_trials(
                X_epoch[selected_flat],
                y_epoch[selected_flat],
                n_imposters=3,
                seed=trial_seed(seed, env, fold, split),
            )
            if set(np.unique(trials["y"]).tolist()) != {0, 1}:
                raise AssertionError("Verification labels must be 1=genuine and 0=impostor")
            trial_sets[(split, env)] = trials
            provenance[(split, env)] = selected_flat[trials["source_local_index"]]

    combined = {}
    for split in ("train", "val"):
        combined[split] = tuple(
            np.concatenate([trial_sets[(split, env)][key] for env in ("pc", "vr")], axis=0)
            for key in ("X", "y", "cid")
        )

    tests = {
        target: split_arrays(data, target.lower(), fold, "test") for target in TARGETS
    }
    for target, (_, labels, claimed) in tests.items():
        if labels.shape[0] != 2016 or int(np.sum(labels == 1)) != 504:
            raise AssertionError(f"Unexpected unchanged {target} test trial counts")
        if set(np.unique(labels).tolist()) != {0, 1} or claimed.shape != labels.shape:
            raise AssertionError(f"Invalid unchanged {target} test labels/claimed IDs")
    return combined, tests, trial_sets, provenance


def save_trial_provenance(
    out_dir: Path,
    fold: int,
    selected: dict,
    trial_sets: dict,
    provenance: dict,
) -> None:
    path = out_dir / "splits" / f"fold_{fold}_mixedmatched_indices.npz"
    with np.load(path, allow_pickle=False) as existing:
        payload = {key: existing[key] for key in existing.files}
    for split in ("train", "val"):
        for env in ("pc", "vr"):
            prefix = f"{split}_{env}"
            payload[f"{prefix}_trial_source_epoch"] = provenance[(split, env)]
            payload[f"{prefix}_trial_label"] = trial_sets[(split, env)]["y"]
            payload[f"{prefix}_trial_claimed_id"] = trial_sets[(split, env)]["cid"]
    np.savez_compressed(path, **payload)


def load_reve(model_dir: Path, device):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(model_dir / "reve-base"), trust_remote_code=True,
        torch_dtype="auto", local_files_only=True,
    ).eval().to(device)
    pos_bank = AutoModel.from_pretrained(
        str(model_dir / "reve-positions"), trust_remote_code=True,
        torch_dtype="auto", local_files_only=True,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in pos_bank.parameters():
        parameter.requires_grad = False
    return model, pos_bank


def evaluate_scores(
    model_name: str,
    target: str,
    fold: int,
    val_scores,
    val_labels,
    test_scores,
    test_labels,
    threshold: float | None,
    seed: int,
) -> dict:
    if threshold is None:
        threshold = threshold_at_validation_eer(val_scores, val_labels)
    metrics = metrics_at_threshold(test_scores, test_labels, threshold)
    return {
        "model": model_name,
        "scenario": f"MixedMatched -> {target}",
        "original_scenario": "MixedMatched",
        "target_environment": target,
        "fold": fold,
        "n_train_pc_per_subject": 39,
        "n_train_vr_per_subject": 39,
        "n_val_pc_per_subject": 9,
        "n_val_vr_per_subject": 9,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "far": metrics["FAR"],
        "frr": metrics["FRR"],
        "eer": compute_eer(test_scores, test_labels),
        "threshold": float(threshold),
        "seed": seed,
        "result_source": "new_mixedmatched_training",
    }


def train_reve_mlp_fold(
    args, meta: dict, fold: int, combined: dict, tests: dict, device, reve, pos_bank
) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from cv5_reve_cache import embed_with_cache
    from head_utils import VerifierHead, train_head
    from verify_utils import collect_scores_labels_head

    Xtr, ytr, cid_tr = combined["train"]
    Xva, yva, cid_va = combined["val"]
    cache = not args.no_cache
    common = f"MixedMatched_fold{fold}"
    Ztr = embed_with_cache(
        reve, pos_bank, Xtr, meta["ch_names"], device, args.reve_bs,
        args.cache_dir, args.model_dir, f"{common}_train", cache,
    )
    Zva = embed_with_cache(
        reve, pos_bank, Xva, meta["ch_names"], device, args.reve_bs,
        args.cache_dir, args.model_dir, f"{common}_val", cache,
    )
    test_embeddings = {
        target: embed_with_cache(
            reve, pos_bank, arrays[0], meta["ch_names"], device, args.reve_bs,
            args.cache_dir, args.model_dir, f"{common}_test_{target.lower()}", cache,
        )
        for target, arrays in tests.items()
    }
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(Ztr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32),
            torch.tensor(cid_tr, dtype=torch.long),
        ),
        batch_size=args.head_bs,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(Zva, dtype=torch.float32),
            torch.tensor(yva, dtype=torch.float32),
            torch.tensor(cid_va, dtype=torch.long),
        ),
        batch_size=args.head_bs,
        shuffle=False,
    )
    n_subjects = len(meta["subject_map"])
    head = VerifierHead(Ztr.shape[1], True, n_subjects).to(device)
    train_head(head, train_loader, val_loader, device, True, args.lr, args.epochs)
    val_scores, val_labels = collect_scores_labels_head(head, val_loader, device, True)
    threshold = threshold_at_validation_eer(val_scores, val_labels)

    rows = []
    for target, (Xte, yte, cid_te) in tests.items():
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(test_embeddings[target], dtype=torch.float32),
                torch.tensor(yte, dtype=torch.float32),
                torch.tensor(cid_te, dtype=torch.long),
            ),
            batch_size=args.head_bs,
            shuffle=False,
        )
        scores, labels = collect_scores_labels_head(head, test_loader, device, True)
        rows.append(
            evaluate_scores(
                "REVE+MLP", target, fold, val_scores, val_labels,
                scores, labels, threshold, args.seed,
            )
        )
    return rows


def eegnet_loader(X, y, cid, n_subjects: int, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from feature_extractors import to_onehot

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(to_onehot(cid, n_subjects), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_eegnet_fold(args, meta: dict, fold: int, combined: dict, tests: dict, device):
    from feature_extractors import (
        build_eegnet_verifier,
        collect_scores_labels_eegnet,
        train_eegnet_end2end,
    )

    Xtr, ytr, cid_tr = combined["train"]
    Xva, yva, cid_va = combined["val"]
    n_subjects = len(meta["subject_map"])
    model = build_eegnet_verifier(16, Xtr.shape[-1], n_subjects, device, True)
    _, threshold = train_eegnet_end2end(
        model, Xtr, ytr, cid_tr, Xva, yva, cid_va,
        device=device, n_subjects=n_subjects, use_claimed_id=True,
        lr=args.lr, epochs=args.epochs, bs=args.head_bs,
    )
    val_loader = eegnet_loader(Xva, yva, cid_va, n_subjects, args.head_bs)
    val_scores, val_labels = collect_scores_labels_eegnet(model, val_loader, device, True)
    rows = []
    for target, (Xte, yte, cid_te) in tests.items():
        test_loader = eegnet_loader(Xte, yte, cid_te, n_subjects, args.head_bs)
        test_scores, test_labels = collect_scores_labels_eegnet(
            model, test_loader, device, True
        )
        rows.append(
            evaluate_scores(
                "EEGNet", target, fold, val_scores, val_labels,
                test_scores, test_labels, float(threshold), args.seed,
            )
        )
    return rows


def baseline_specs() -> list[dict]:
    return [
        {"scenario": "Within -> PC", "original": "PC->PC", "target": "PC", "counts": (78, 0, 18, 0)},
        {"scenario": "Cross -> PC", "original": "VR->PC", "target": "PC", "counts": (0, 78, 0, 18)},
        {"scenario": "MixedFull -> PC", "original": "Mixed->PC", "target": "PC", "counts": (78, 78, 18, 18)},
        {"scenario": "Within -> VR", "original": "VR->VR", "target": "VR", "counts": (0, 78, 0, 18)},
        {"scenario": "Cross -> VR", "original": "PC->VR", "target": "VR", "counts": (78, 0, 18, 0)},
        {"scenario": "MixedFull -> VR", "original": "Mixed->VR", "target": "VR", "counts": (78, 78, 18, 18)},
    ]


def load_existing_rows(path: Path, models: list[str], seed: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corrected CV5 per-fold results: {path}")
    frame = pd.read_csv(path)
    required = {
        "model", "scenario", "fold", "accuracy", "balanced_accuracy",
        "FAR", "FRR", "EER", "threshold",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Existing result CSV is missing columns: {missing}")

    rows = []
    for model in models:
        for spec in baseline_specs():
            subset = frame[
                (frame["model"] == model) & (frame["scenario"] == spec["original"])
            ].sort_values("fold")
            if subset["fold"].tolist() != list(range(5)):
                raise AssertionError(
                    f"Expected exactly folds 0..4 for {model}, {spec['original']}"
                )
            train_pc, train_vr, val_pc, val_vr = spec["counts"]
            for record in subset.to_dict("records"):
                rows.append(
                    {
                        "model": model,
                        "scenario": spec["scenario"],
                        "original_scenario": spec["original"],
                        "target_environment": spec["target"],
                        "fold": int(record["fold"]),
                        "n_train_pc_per_subject": train_pc,
                        "n_train_vr_per_subject": train_vr,
                        "n_val_pc_per_subject": val_pc,
                        "n_val_vr_per_subject": val_vr,
                        "accuracy": float(record["accuracy"]),
                        "balanced_accuracy": float(record["balanced_accuracy"]),
                        "far": float(record["FAR"]),
                        "frr": float(record["FRR"]),
                        "eer": float(record["EER"]),
                        "threshold": float(record["threshold"]),
                        "seed": int(record.get("seed", seed)),
                        "result_source": str(path.resolve()),
                    }
                )
    return rows


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scenario), group in frame.groupby(["model", "scenario"], sort=False):
        if sorted(group["fold"].tolist()) != list(range(5)):
            raise AssertionError(f"Expected exactly five folds for {model}, {scenario}")
        row = {"model": model, "scenario": scenario, "n_folds": 5}
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_and_recovery(frame: pd.DataFrame):
    pairwise_rows = []
    recovery_rows = []
    for model in MODELS:
        if model not in set(frame["model"]):
            continue
        for target in TARGETS:
            indexed = {
                name: frame[
                    (frame["model"] == model)
                    & (frame["scenario"] == f"{name} -> {target}")
                ].sort_values("fold")
                for name in ("Cross", "MixedMatched", "MixedFull")
            }
            for name, subset in indexed.items():
                if subset["fold"].tolist() != list(range(5)):
                    raise AssertionError(f"Missing paired folds for {model}, {name} -> {target}")

            for metric in ("frr", "eer"):
                for first, second in (("Cross", "MixedMatched"), ("MixedMatched", "MixedFull")):
                    a = indexed[first][metric].to_numpy(dtype=float)
                    b = indexed[second][metric].to_numpy(dtype=float)
                    difference = a - b
                    if np.allclose(difference, 0.0):
                        statistic, p_value = 0.0, 1.0
                    else:
                        test = stats.wilcoxon(a, b, alternative="two-sided", method="auto")
                        statistic, p_value = float(test.statistic), float(test.pvalue)
                    pairwise_rows.append(
                        {
                            "model": model,
                            "target_environment": target,
                            "metric": metric.upper(),
                            "comparison": f"{first} - {second}",
                            "first_mean": float(a.mean()),
                            "second_mean": float(b.mean()),
                            "delta_mean_first_minus_second": float(difference.mean()),
                            "absolute_delta_mean": float(abs(difference.mean())),
                            "delta_sd": float(difference.std(ddof=1)),
                            "wilcoxon_statistic": statistic,
                            "raw_p_value": p_value,
                            "n_paired_folds": 5,
                            "inference_scope": "descriptive paired fold-level comparison",
                        }
                    )

                cross = float(indexed["Cross"][metric].mean())
                matched = float(indexed["MixedMatched"][metric].mean())
                full = float(indexed["MixedFull"][metric].mean())
                denominator = cross - full
                recovery = np.nan if np.isclose(denominator, 0.0) else (cross - matched) / denominator
                recovery_rows.append(
                    {
                        "model": model,
                        "target_environment": target,
                        "metric": metric.upper(),
                        "cross_mean": cross,
                        "mixedmatched_mean": matched,
                        "mixedfull_mean": full,
                        "cross_minus_mixedmatched": cross - matched,
                        "mixedmatched_minus_mixedfull": matched - full,
                        "recovery_fraction": recovery,
                    }
                )
    return pd.DataFrame(pairwise_rows), pd.DataFrame(recovery_rows)


def print_summary(summary: pd.DataFrame, recovery: pd.DataFrame) -> None:
    for model in MODELS:
        model_rows = summary[summary["model"] == model]
        if model_rows.empty:
            continue
        print(f"\n{model}")
        for target in TARGETS:
            print(f"{target} test:")
            for condition in ("Cross", "MixedMatched", "MixedFull"):
                row = model_rows[model_rows["scenario"] == f"{condition} -> {target}"].iloc[0]
                print(
                    f"  {condition}: FRR={100*row['frr_mean']:.2f}% "
                    f"(+/- {100*row['frr_sd']:.2f}), EER={100*row['eer_mean']:.2f}% "
                    f"(+/- {100*row['eer_sd']:.2f})"
                )
            for metric in ("FRR", "EER"):
                value = recovery[
                    (recovery["model"] == model)
                    & (recovery["target_environment"] == target)
                    & (recovery["metric"] == metric)
                ]["recovery_fraction"].iloc[0]
                print(f"  {metric} recovery fraction: {value:.4f}")


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    existing_path = Path(args.existing_results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))
    folds = json.loads(str(data["folds_json"]))
    subjects = sorted(int(value) for value in meta["subject_map"].values())
    if len(subjects) != 21 or int(meta["n_folds"]) != 5:
        raise AssertionError("MixedMatched requires exactly 21 subjects and five folds")
    if int(meta.get("n_imposters", 3)) != 3:
        raise AssertionError("MixedMatched requires n_imposters=3")

    fold_data = {}
    sanity_rows = []
    for fold in range(5):
        selected = select_matched_indices(folds, subjects, fold, args.seed)
        sanity_rows.extend(
            validate_and_save_splits(
                data, folds, subjects, fold, selected, out_dir, args.seed
            )
        )
        combined, tests, trial_sets, provenance = build_trials_for_fold(
            data, subjects, fold, selected, args.seed
        )
        save_trial_provenance(out_dir, fold, selected, trial_sets, provenance)
        fold_data[fold] = (combined, tests)

    sanity = pd.DataFrame(sanity_rows)
    sanity.to_csv(out_dir / "mixedmatched_split_sanity.csv", index=False)
    sanity_summary = {
        "subjects": 21,
        "folds": 5,
        "rows": int(len(sanity)),
        "train_pc_per_subject": 39,
        "train_vr_per_subject": 39,
        "train_total_per_subject": 78,
        "validation_pc_per_subject": 9,
        "validation_vr_per_subject": 9,
        "validation_total_per_subject": 18,
        "test_per_subject_target_environment": 24,
        "all_no_duplicates": bool(sanity["no_duplicates"].all()),
        "all_no_overlap": bool(sanity["no_overlap"].all()),
        "all_same_original_test_fold": bool(sanity["same_original_test_fold"].all()),
        "all_deterministic_rerun": bool(sanity["deterministic_rerun"].all()),
    }
    (out_dir / "mixedmatched_split_sanity.json").write_text(
        json.dumps(sanity_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(sanity_summary, indent=2))
    if args.prepare_only:
        print("Prepared and validated MixedMatched subsets; --prepare_only requested.")
        return

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = list(MODELS) if args.model == "all" else [args.model]
    trained_path = out_dir / "mixedmatched_trained_per_fold.csv"
    if trained_path.is_file():
        trained_rows = pd.read_csv(trained_path).to_dict("records")
    else:
        trained_rows = []
    if args.overwrite:
        trained_rows = [row for row in trained_rows if row["model"] not in models]
    completed = {
        (row["model"], int(row["fold"]), row["target_environment"])
        for row in trained_rows
    }

    need_reve = "REVE+MLP" in models and any(
        ("REVE+MLP", fold, target) not in completed
        for fold in range(5) for target in TARGETS
    )
    reve, pos_bank = load_reve(Path(args.model_dir), device) if need_reve else (None, None)

    for model in models:
        for fold in range(5):
            if all((model, fold, target) in completed for target in TARGETS):
                print(f"[SKIP] {model} fold {fold}: both target rows already saved")
                continue
            set_seed(args.seed + fold)
            combined, tests = fold_data[fold]
            print(f"[RUN] {model} | MixedMatched | fold {fold} | PC and VR tests")
            if model == "REVE+MLP":
                new_rows = train_reve_mlp_fold(
                    args, meta, fold, combined, tests, device, reve, pos_bank
                )
            else:
                new_rows = train_eegnet_fold(args, meta, fold, combined, tests, device)
            trained_rows = [
                row for row in trained_rows
                if not (row["model"] == model and int(row["fold"]) == fold)
            ]
            trained_rows.extend(new_rows)
            pd.DataFrame(trained_rows).to_csv(trained_path, index=False)
            completed.update((model, fold, target) for target in TARGETS)

    trained = pd.DataFrame(trained_rows)
    expected_new = len(models) * 5 * 2
    selected_trained = trained[trained["model"].isin(models)]
    if len(selected_trained) != expected_new:
        raise AssertionError(f"Expected {expected_new} MixedMatched rows, found {len(selected_trained)}")

    existing_rows = load_existing_rows(existing_path, models, args.seed)
    combined_results = pd.concat(
        [pd.DataFrame(existing_rows), selected_trained], ignore_index=True
    ).sort_values(["model", "target_environment", "scenario", "fold"])
    results_path = out_dir / "mixed_size_control_results.csv"
    combined_results.to_csv(results_path, index=False)
    summary = summarize_results(combined_results)
    summary_path = out_dir / "mixed_size_control_summary.csv"
    summary.to_csv(summary_path, index=False)
    pairwise, recovery = pairwise_and_recovery(combined_results)
    pairwise.to_csv(out_dir / "mixed_size_control_pairwise.csv", index=False)
    recovery.to_csv(out_dir / "mixed_size_control_recovery.csv", index=False)

    manifest = {
        "source_cv5_npz": str(npz_path.resolve()),
        "source_cv5_npz_sha256": file_sha256(npz_path),
        "existing_results": str(existing_path.resolve()),
        "existing_results_sha256": file_sha256(existing_path),
        "existing_results_modified": False,
        "models": models,
        "claimed_identity_used": True,
        "n_imposters": 3,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "head_batch_size": args.head_bs,
        "reve_batch_size": args.reve_bs,
        "device": str(device),
        "mixedmatched_model_fits": len(models) * 5,
        "each_fit_evaluated_on": ["PC", "VR"],
        "metrics_threshold": "validation EER threshold only",
        "test_eer": "computed independently from test scores",
    }
    (out_dir / "mixed_size_control_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print_summary(summary, recovery)
    print(f"\nSaved fold results: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()


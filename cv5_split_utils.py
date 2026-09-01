"""Deterministic 5-fold split and trial construction utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ENV_OFFSETS = {"pc": 100_000, "vr": 200_000}
SPLIT_OFFSETS = {"train": 10, "val": 20, "test": 30}


@dataclass(frozen=True)
class FoldConfig:
    n_folds: int = 5
    epochs_per_subject_env: int = 120
    test_epochs: int = 24
    train_epochs: int = 78
    val_epochs: int = 18
    seed: int = 42
    n_imposters: int = 3


def generate_epoch_fold_indices(y_by_env: dict[str, np.ndarray], config: FoldConfig):
    """Create auditable per-env/per-subject epoch indices for every fold."""
    folds: dict[str, dict[int, dict[int, dict[str, list[int]]]]] = {}

    for env, y in y_by_env.items():
        if env not in ENV_OFFSETS:
            raise ValueError(f"Unsupported env: {env}")
        folds[env] = {}
        for subj in sorted(np.unique(y).astype(int).tolist()):
            subj_idx = np.where(y == subj)[0].astype(np.int64)
            if subj_idx.size != config.epochs_per_subject_env:
                raise ValueError(
                    f"{env} subject {subj}: expected {config.epochs_per_subject_env} epochs, "
                    f"found {subj_idx.size}"
                )

            outer_seed = config.seed + ENV_OFFSETS[env] + subj * 1_000
            rng_outer = np.random.default_rng(outer_seed)
            shuffled = subj_idx.copy()
            rng_outer.shuffle(shuffled)
            test_chunks = np.split(shuffled, config.n_folds)

            folds[env][subj] = {}
            for fold in range(config.n_folds):
                test_idx = np.sort(test_chunks[fold])
                remaining = np.setdiff1d(subj_idx, test_idx, assume_unique=False)
                inner_seed = config.seed + ENV_OFFSETS[env] + subj * 1_000 + fold * 100 + 7
                rng_inner = np.random.default_rng(inner_seed)
                remaining = remaining.copy()
                rng_inner.shuffle(remaining)

                train_idx = np.sort(remaining[: config.train_epochs])
                val_idx = np.sort(remaining[config.train_epochs : config.train_epochs + config.val_epochs])
                if train_idx.size != config.train_epochs or val_idx.size != config.val_epochs:
                    raise RuntimeError(f"Bad split sizes for {env} subject {subj} fold {fold}")

                folds[env][subj][fold] = {
                    "train": train_idx.astype(int).tolist(),
                    "val": val_idx.astype(int).tolist(),
                    "test": test_idx.astype(int).tolist(),
                    "outer_seed": int(outer_seed),
                    "inner_seed": int(inner_seed),
                }

    return folds


def validate_epoch_folds(folds, config: FoldConfig):
    """Check that every subject/env epoch appears in exactly one test fold."""
    for env, subj_map in folds.items():
        for subj, fold_map in subj_map.items():
            test_all = []
            for fold in range(config.n_folds):
                split = fold_map[fold]
                tr = set(split["train"])
                va = set(split["val"])
                te = set(split["test"])
                if tr & va or tr & te or va & te:
                    raise ValueError(f"Overlap in {env} subject {subj} fold {fold}")
                test_all.extend(split["test"])
            if len(test_all) != config.epochs_per_subject_env:
                raise ValueError(f"Incorrect total test count for {env} subject {subj}")
            vals, counts = np.unique(test_all, return_counts=True)
            if not np.all(counts == 1):
                raise ValueError(f"Some test epochs are repeated for {env} subject {subj}")


def save_fold_indices(folds, out_dir: Path, config: FoldConfig):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cv5_epoch_indices.json"
    csv_path = out_dir / "cv5_epoch_indices.csv"

    serializable = {
        "config": config.__dict__,
        "folds": {
            env: {
                str(subj): {str(fold): split for fold, split in fold_map.items()}
                for subj, fold_map in subj_map.items()
            }
            for env, subj_map in folds.items()
        },
    }
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    rows = []
    for env, subj_map in folds.items():
        for subj, fold_map in subj_map.items():
            for fold, split_map in fold_map.items():
                for split_name in ["train", "val", "test"]:
                    for idx in split_map[split_name]:
                        rows.append(
                            {
                                "env": env,
                                "subject": subj,
                                "fold": fold,
                                "split": split_name,
                                "epoch_index": idx,
                                "outer_seed": split_map["outer_seed"],
                                "inner_seed": split_map["inner_seed"],
                            }
                        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def gather_epoch_split(X, y, folds, env: str, fold: int, split: str):
    idx = []
    for subj in sorted(np.unique(y).astype(int).tolist()):
        idx.extend(folds[env][subj][fold][split])
    idx = np.asarray(idx, dtype=np.int64)
    return X[idx], y[idx], idx


def to_bct(X_tbc):
    return np.transpose(X_tbc, (0, 2, 1)).astype(np.float32)


def supervised_trials(X, y, claimed_source_y=None, n_imposters=3, seed=42):
    """Create 1 genuine + n_imposters zero-effort impostor trials per genuine epoch."""
    rng = np.random.default_rng(seed)
    if claimed_source_y is None:
        claimed_source_y = y

    trials, labels, claimed_ids, source_indices = [], [], [], []
    for subj in sorted(np.unique(claimed_source_y).astype(int).tolist()):
        subj_idx = np.where(y == subj)[0]
        imp_idx = np.where(y != subj)[0]
        if imp_idx.size < n_imposters:
            raise ValueError(f"Not enough impostors for subject {subj}")
        for local_i in subj_idx:
            trials.append(X[local_i])
            labels.append(1)
            claimed_ids.append(subj)
            source_indices.append(local_i)
            for local_j in rng.choice(imp_idx, size=n_imposters, replace=False):
                trials.append(X[local_j])
                labels.append(0)
                claimed_ids.append(subj)
                source_indices.append(local_j)

    return {
        "X": to_bct(np.asarray(trials, dtype=np.float32)),
        "y": np.asarray(labels, dtype=np.int32),
        "cid": np.asarray(claimed_ids, dtype=np.int32),
        "source_local_index": np.asarray(source_indices, dtype=np.int64),
    }


def oneclass_subject_sets(X, y, folds, env: str, fold: int, subject: int):
    tr_idx = np.asarray(folds[env][subject][fold]["train"], dtype=np.int64)
    va_g_idx = np.asarray(folds[env][subject][fold]["val"], dtype=np.int64)
    te_g_idx = np.asarray(folds[env][subject][fold]["test"], dtype=np.int64)

    va_i_idx = []
    te_i_idx = []
    for other in sorted(np.unique(y).astype(int).tolist()):
        if other == subject:
            continue
        va_i_idx.extend(folds[env][other][fold]["val"])
        te_i_idx.extend(folds[env][other][fold]["test"])

    va_i_idx = np.asarray(va_i_idx, dtype=np.int64)
    te_i_idx = np.asarray(te_i_idx, dtype=np.int64)
    Xva = np.concatenate([X[va_g_idx], X[va_i_idx]], axis=0)
    yva = np.concatenate([np.ones(va_g_idx.size, dtype=np.int32), np.zeros(va_i_idx.size, dtype=np.int32)])
    Xte = np.concatenate([X[te_g_idx], X[te_i_idx]], axis=0)
    yte = np.concatenate([np.ones(te_g_idx.size, dtype=np.int32), np.zeros(te_i_idx.size, dtype=np.int32)])
    return {
        "Xtr": to_bct(X[tr_idx]),
        "Xva": to_bct(Xva),
        "yva": yva,
        "Xte": to_bct(Xte),
        "yte": yte,
        "train_indices": tr_idx,
        "val_genuine_indices": va_g_idx,
        "val_impostor_indices": va_i_idx,
        "test_genuine_indices": te_g_idx,
        "test_impostor_indices": te_i_idx,
    }


def trial_seed(base_seed: int, scenario: str, fold: int, split: str):
    scenario_value = sum((i + 1) * ord(ch) for i, ch in enumerate(scenario))
    return int(base_seed + 500_000 + scenario_value + fold * 1_000 + SPLIT_OFFSETS[split])

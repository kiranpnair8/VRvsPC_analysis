#!/usr/bin/env python3
"""Build deterministic 5-fold PC/VR verification datasets.

This is additive and does not replace build_verification_dataset.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cv5_split_utils import (
    FoldConfig,
    gather_epoch_split,
    generate_epoch_fold_indices,
    oneclass_subject_sets,
    save_fold_indices,
    supervised_trials,
    trial_seed,
    validate_epoch_folds,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source_npz",
        default="./out/pc_vr_verification_dataset_lphp10_50.npz",
        help="Existing preprocessed verification NPZ to reconstruct epoch-level CV data from",
    )
    ap.add_argument("--out_dir", default="./out/cv5", help="Output directory for CV datasets and indices")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_imposters", type=int, default=3)
    ap.add_argument("--epochs_per_subject_env", type=int, default=120)
    return ap.parse_args()


def load_source_meta(source):
    if "meta_json" not in source:
        return {}
    value = source["meta_json"]
    if hasattr(value, "item"):
        value = value.item()
    return json.loads(str(value))


def as_epoch_tbc(X, ch_names):
    """Convert source verification trials from [B,C,T] to internal [B,T,C]."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D X array, found shape {X.shape}")

    n_channels = len(ch_names)
    if X.shape[1] == n_channels:
        return np.transpose(X, (0, 2, 1)).astype(np.float32)
    if X.shape[2] == n_channels:
        return X.astype(np.float32)
    raise ValueError(f"Cannot infer channel axis for shape {X.shape} and {n_channels} channels")


def reconstruct_env_epochs(source, env, ch_names, expected_total):
    split_specs = [("tr", 78), ("va", 18), ("te", 24)]
    split_arrays = {}
    subjects = set()

    for split, _ in split_specs:
        y = np.asarray(source[f"y{split}_{env}"], dtype=np.int32)
        cid = np.asarray(source[f"cid_{split}_{env}"], dtype=np.int32)
        X = as_epoch_tbc(source[f"X{split}_{env}"], ch_names)
        subjects.update(cid[y == 1].astype(int).tolist())
        split_arrays[split] = (X, y, cid)

    X_all, y_all = [], []
    reconstruction_rows = []
    for subject in sorted(subjects):
        pieces = []
        for split, expected_split_count in split_specs:
            X, y, cid = split_arrays[split]
            mask = (y == 1) & (cid == subject)
            count = int(mask.sum())
            if count != expected_split_count:
                raise ValueError(
                    f"{env} subject {subject} split {split}: expected {expected_split_count} "
                    f"genuine epochs, found {count}"
                )
            pieces.append(X[mask])
            reconstruction_rows.append(
                {
                    "env": env,
                    "subject": subject,
                    "source_split": split,
                    "genuine_epochs": count,
                }
            )

        X_subject = np.concatenate(pieces, axis=0)
        if X_subject.shape[0] != expected_total:
            raise ValueError(
                f"{env} subject {subject}: expected {expected_total} reconstructed genuine epochs, "
                f"found {X_subject.shape[0]}"
            )
        X_all.append(X_subject)
        y_all.append(np.full((X_subject.shape[0],), subject, dtype=np.int32))

    if not X_all:
        raise ValueError(f"No genuine epochs reconstructed for {env}")
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), reconstruction_rows


def main():
    args = parse_args()
    source_npz = Path(args.source_npz)
    if not source_npz.exists():
        raise FileNotFoundError(f"source_npz does not exist: {source_npz}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = np.load(source_npz, allow_pickle=True)
    source_meta = load_source_meta(source)
    ch_names = source_meta.get(
        "ch_names",
        ["Fp1", "Fp2", "Fc5", "Fz", "Fc6", "T7", "Cz", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"],
    )
    subject_map = source_meta.get("subject_map", {})
    X_pc, y_pc, pc_reconstruction = reconstruct_env_epochs(
        source,
        "pc",
        ch_names,
        args.epochs_per_subject_env,
    )
    X_vr, y_vr, vr_reconstruction = reconstruct_env_epochs(
        source,
        "vr",
        ch_names,
        args.epochs_per_subject_env,
    )

    config = FoldConfig(
        seed=args.seed,
        n_imposters=args.n_imposters,
        epochs_per_subject_env=args.epochs_per_subject_env,
    )

    folds = generate_epoch_fold_indices({"pc": y_pc, "vr": y_vr}, config)
    validate_epoch_folds(folds, config)
    indices_json, indices_csv = save_fold_indices(folds, out_dir, config)

    arrays = {
        "X_epoch_pc": X_pc.astype(np.float32),
        "y_epoch_pc": y_pc.astype(np.int32),
        "X_epoch_vr": X_vr.astype(np.float32),
        "y_epoch_vr": y_vr.astype(np.int32),
    }

    supervised_index_rows = []
    oneclass_data = {"pc": {}, "vr": {}}

    for fold in range(config.n_folds):
        for env, X, y in [("pc", X_pc, y_pc), ("vr", X_vr, y_vr)]:
            oneclass_data[env][fold] = {}
            for split in ["train", "val", "test"]:
                X_split, y_split, epoch_idx = gather_epoch_split(X, y, folds, env, fold, split)
                seed = trial_seed(args.seed, env, fold, split)
                trials = supervised_trials(
                    X_split,
                    y_split,
                    n_imposters=args.n_imposters,
                    seed=seed,
                )
                prefix = f"{env}_fold{fold}_{split}"
                arrays[f"X_{prefix}"] = trials["X"]
                arrays[f"y_{prefix}"] = trials["y"]
                arrays[f"cid_{prefix}"] = trials["cid"]
                arrays[f"trial_source_epoch_{prefix}"] = epoch_idx[trials["source_local_index"]].astype(np.int64)

                for trial_i, source_epoch in enumerate(arrays[f"trial_source_epoch_{prefix}"]):
                    supervised_index_rows.append(
                        {
                            "env": env,
                            "fold": fold,
                            "split": split,
                            "trial_index": trial_i,
                            "source_epoch_index": int(source_epoch),
                            "label": int(trials["y"][trial_i]),
                            "claimed_id": int(trials["cid"][trial_i]),
                            "seed": seed,
                        }
                    )

            for subj in sorted(np.unique(y).astype(int).tolist()):
                oneclass_data[env][fold][subj] = oneclass_subject_sets(X, y, folds, env, fold, subj)

    import pandas as pd

    trial_csv = out_dir / "cv5_supervised_trial_indices.csv"
    pd.DataFrame(supervised_index_rows).to_csv(trial_csv, index=False)

    meta = {
        "source": "reconstructed_from_preprocessed_verification_npz_genuine_trials_only",
        "source_npz": str(source_npz),
        "source_dataset_name": source_npz.name,
        "preprocessing": "No filtering, baseline correction, z-scoring, or resampling was applied by this CV5 builder.",
        "source_reconstruction": pc_reconstruction + vr_reconstruction,
        "fs": source_meta.get("fs"),
        "epoch_window_sec": source_meta.get("epoch_window_sec"),
        "epoch_samples": int(X_pc.shape[1]),
        "notch_freq": source_meta.get("notch_freq"),
        "band": source_meta.get("band"),
        "filter_order": source_meta.get("filter_order"),
        "split_counts": {
            "train": config.train_epochs,
            "val": config.val_epochs,
            "test": config.test_epochs,
        },
        "split_percentages": {"train": 0.65, "val": 0.15, "test": 0.20},
        "n_folds": config.n_folds,
        "n_imposters": args.n_imposters,
        "seed": args.seed,
        "subject_map": subject_map,
        "ch_names": ch_names,
        "fold_indices_json": str(indices_json),
        "fold_indices_csv": str(indices_csv),
        "supervised_trial_indices_csv": str(trial_csv),
        "format": "supervised X arrays are [B,C,T], epoch X arrays are [B,T,C]",
    }

    out_path = out_dir / "cv5_verification_dataset_lphp10_50.npz"
    np.savez_compressed(
        out_path,
        **arrays,
        oneclass=oneclass_data,
        folds_json=json.dumps(
            {
                env: {
                    str(subj): {str(fold): split for fold, split in fold_map.items()}
                    for subj, fold_map in subj_map.items()
                }
                for env, subj_map in folds.items()
            }
        ),
        meta_json=json.dumps(meta),
    )
    print(f"Saved CV5 dataset: {out_path}")
    print(f"Reconstructed genuine epochs from: {source_npz}")
    print(f"Saved fold indices: {indices_json}")
    print(f"Saved supervised trial indices: {trial_csv}")


if __name__ == "__main__":
    main()

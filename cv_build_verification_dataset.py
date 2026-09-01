#!/usr/bin/env python3
"""Build deterministic 5-fold PC/VR verification datasets.

This is additive and does not replace build_verification_dataset.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_verification_dataset import (
    BAND_HIGH,
    BAND_LOW,
    CHANNELS,
    FILTER_ORDER,
    FS,
    NOTCH_FREQ,
    POST_SAMPLES,
    PRE_SAMPLES,
    combine_epochs_by_subject,
)
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
    ap.add_argument("--data_dir", required=True, help="Folder containing Subject_XX_PC.csv and Subject_XX_VR.csv")
    ap.add_argument("--out_dir", default="./out/cv5", help="Output directory for CV datasets and indices")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_imposters", type=int, default=3)
    ap.add_argument("--epochs_per_subject_env", type=int, default=120)
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_pc, y_pc, X_vr, y_vr, subject_map = combine_epochs_by_subject(data_dir)
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
        "fs": FS,
        "epoch_window_sec": [-(PRE_SAMPLES / FS), POST_SAMPLES / FS],
        "epoch_samples": PRE_SAMPLES + POST_SAMPLES,
        "notch_freq": NOTCH_FREQ,
        "band": [BAND_LOW, BAND_HIGH],
        "filter_order": FILTER_ORDER,
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
        "ch_names": CHANNELS,
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
    print(f"Saved fold indices: {indices_json}")
    print(f"Saved supervised trial indices: {trial_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build and save PC/VR P300 verification datasets (genuine vs impostor) locally.

What it does
------------
1) Reads per-subject CSV files from a folder (e.g., Subject_01_PC.csv, Subject_01_VR.csv)
2) Filters continuous EEG (60 Hz notch + 0.1–30 Hz bandpass)
3) Extracts target-locked epochs (-200ms to +800ms) for each file
4) Baseline-corrects + epoch-wise z-score normalization (per channel)
5) Builds pooled epoch datasets separately for PC and VR: X_pc, y_pc, X_vr, y_vr
6) Splits epochs per subject into train/val/test (defaults 0.65/0.15/0.20)
7) Expands each split into verification trials:
      For each epoch from a subject:
        - 1 genuine trial (label=1, claimed_id=subject)
        - n_imposters impostor trials (label=0, claimed_id=subject), drawn from other subjects
8) Transposes to REVE-friendly format: (Batch, Channels, Time)
9) Saves everything to a compressed .npz file.

Run
---
python build_verification_dataset.py --data_dir "G:/Shared drives/VR-BCI/Visual P300 in VR vs PC/src/Data" --out_dir "./out"

Outputs
-------
out/pc_vr_verification_dataset.npz

Notes
-----
- This script is for the *Colab-style verification dataset* (trials + claimed_id),
  NOT your EER/template protocol.
- Colab paths (/content/drive/...) will NOT work locally.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


# -----------------------------
# Signal processing config
# -----------------------------
NOTCH_FREQ = 60
BAND_LOW = 10
BAND_HIGH = 50
FILTER_ORDER = 4
FS = 512
PRE_SAMPLES = int(0.2 * FS)   # 200 ms
POST_SAMPLES = int(0.8 * FS)  # 800 ms


CHANNELS = ["Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"]




# -----------------------------
# Filtering
# -----------------------------
def bandpass_filter(signal: np.ndarray, fs: int, low=0.1, high=30, order=4) -> np.ndarray:
    """signal: [N, C]"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, fs: int, freq=60, q=30) -> np.ndarray:
    """signal: [N, C]"""
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal, axis=0)


# -----------------------------
# Epoch extraction
# -----------------------------
def extract_p300_epochs(file_path: Path) -> np.ndarray:
    """
    Reads one CSV continuous EEG file (NO HEADER) and returns epochs:
      epochs: [n_epochs, time, channels]
    Assumes fixed layout:
      col 0  = Time
      col 1..16 = 16 EEG channels
      col 17 = Event
      col 18 = IsTarget
      (others may exist after that)
    """
    df = pd.read_csv(file_path, header=None, sep=None, engine="python")

    # Need at least 19 columns to access [1:17] and [18]
    if df.shape[1] < 19:
        raise ValueError(f"{file_path}: expected >= 19 columns, got {df.shape[1]}")

    # Extract EEG + IsTarget, coerce numeric (bad strings/blanks -> NaN)
    ch = df.iloc[:, 1:17].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)  # [N,16]
    isTarget = pd.to_numeric(df.iloc[:, 18], errors="coerce").to_numpy(dtype=np.float32)    # [N]

    # Drop rows where ANY channel or IsTarget is NaN/Inf (row-wise cleanup)
    good = np.isfinite(ch).all(axis=1) & np.isfinite(isTarget)
    ch = ch[good]
    isTarget = isTarget[good].astype(np.int32)

    # Filtering on continuous stream (now safe: no NaNs)
    ch = notch_filter(ch, FS, freq=NOTCH_FREQ)
    ch = bandpass_filter(ch, FS, BAND_LOW, BAND_HIGH, FILTER_ORDER)

    stimulus_indices = np.where(isTarget == 1)[0]

    epochs = []
    for i in stimulus_indices:
        start = i - PRE_SAMPLES
        end = i + POST_SAMPLES  # end excluded

        if start >= 0 and end < len(ch):
            epoch = ch[start:end, :]  # [T,16]

            # Baseline correction
            baseline = epoch[:PRE_SAMPLES].mean(axis=0, keepdims=True)
            epoch = epoch - baseline

            # Epoch-wise z-score normalization
            mean = epoch.mean(axis=0, keepdims=True)
            std = epoch.std(axis=0, keepdims=True) + 1e-6
            epoch = (epoch - mean) / std

            epochs.append(epoch)

    if len(epochs) == 0:
        return np.empty((0, PRE_SAMPLES + POST_SAMPLES, 16), dtype=np.float32)

    return np.asarray(epochs, dtype=np.float32)


# -----------------------------
# Combine epochs per subject, separately for PC and VR
# -----------------------------
def combine_epochs_by_subject(data_dir: Path):
    """
    Returns:
      X_pc: [N_pc, T, C], y_pc: [N_pc]
      X_vr: [N_vr, T, C], y_vr: [N_vr]
      subject_map: dict subject_id -> integer label
    Expects filenames like: Subject_01_PC.csv or Subject_01_VR.csv
    """
    X_pc, y_pc = [], []
    X_vr, y_vr = [], []

    subject_map = {}
    next_label = 0

    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".csv"])

    for file_path in files:
        file = file_path.name
        parts = file.replace(".csv", "").split("_")
        if len(parts) < 3:
            print(f"[WARN] Skipping unexpected filename: {file}")
            continue

        subject_id = parts[0] + "_" + parts[1]  # e.g., Subject_01
        condition = parts[2].upper()            # PC or VR

        if subject_id not in subject_map:
            subject_map[subject_id] = next_label
            next_label += 1

        label = subject_map[subject_id]

        epochs = extract_p300_epochs(file_path)  # [nE, T, 16]
        if epochs.shape[0] == 0:
            print(f"[WARN] No epochs extracted: {file}")
            continue

        if condition == "PC":
            X_pc.append(epochs)
            y_pc.append(np.full((epochs.shape[0],), label, dtype=np.int32))
        elif condition == "VR":
            X_vr.append(epochs)
            y_vr.append(np.full((epochs.shape[0],), label, dtype=np.int32))
        else:
            print(f"[WARN] Unknown condition in filename: {file} (got {condition})")
            continue

        print(f"{file}: epochs {epochs.shape}")

    if len(X_pc) == 0 or len(X_vr) == 0:
        raise RuntimeError(
            f"Did not find enough data. Found PC files: {len(X_pc)} chunks, VR files: {len(X_vr)} chunks.\n"
            f"Check your folder and naming convention."
        )

    X_pc = np.concatenate(X_pc, axis=0)
    y_pc = np.concatenate(y_pc, axis=0)
    X_vr = np.concatenate(X_vr, axis=0)
    y_vr = np.concatenate(y_vr, axis=0)

    return X_pc, y_pc, X_vr, y_vr, subject_map


# -----------------------------
# Split per subject into train/val/test
# -----------------------------
def split_epochs_per_subject(X: np.ndarray, y: np.ndarray, train=0.65, val=0.15, seed=42):
    """
    X: [N, T, C], y: [N]
    Returns: Xtr,ytr,Xva,yva,Xte,yte
    Splits *within each subject*, then concatenates indices across subjects.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []

    for subj in np.unique(y):
        idx = np.where(y == subj)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_tr = int(train * n)
        n_va = int(val * n)

        tr.extend(idx[:n_tr])
        va.extend(idx[n_tr:n_tr + n_va])
        te.extend(idx[n_tr + n_va:])

    tr = np.array(tr, dtype=np.int64)
    va = np.array(va, dtype=np.int64)
    te = np.array(te, dtype=np.int64)

    return X[tr], y[tr], X[va], y[va], X[te], y[te]


# -----------------------------
# Build genuine/impostor trials
# -----------------------------
def genuine_imposter_trials(X: np.ndarray, y: np.ndarray, n_imposters=3, seed=42):
    """
    For each epoch of each subject:
      - one genuine trial (label=1, claimed_id=subj)
      - n_imposters impostor trials drawn from other subjects (label=0, claimed_id=subj)

    Returns:
      trials: [N_trials, T, C]
      labels: [N_trials]  (1 genuine, 0 impostor)
      claimed_ids: [N_trials]  (subject label being claimed)
    """
    rng = np.random.default_rng(seed)

    trials, labels, claimed_ids = [], [], []

    for subj in np.unique(y):
        subj_idx = np.where(y == subj)[0]
        imp_idx = np.where(y != subj)[0]

        if len(imp_idx) < n_imposters:
            raise RuntimeError(f"Not enough impostor samples for subj={subj} (need {n_imposters}, have {len(imp_idx)})")

        for i in subj_idx:
            # genuine
            trials.append(X[i])
            labels.append(1)
            claimed_ids.append(subj)

            # impostors
            imposters = rng.choice(imp_idx, n_imposters, replace=False)
            for j in imposters:
                trials.append(X[j])
                labels.append(0)
                claimed_ids.append(subj)

    return (
        np.asarray(trials, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(claimed_ids, dtype=np.int32),
    )


def to_reve_format(X: np.ndarray) -> np.ndarray:
    """
    Convert from [B, T, C] to [B, C, T]
    """
    return np.transpose(X, (0, 2, 1)).astype(np.float32)

def build_oneclass_splits(X, y, subject_id):
    """
    X: [N, T, C]
    y: [N]
    subject_id: int

    Returns:
      Xtr, Xva, Xte, yte
    """
    # Genuine samples
    idx_g = np.where(y == subject_id)[0]

    # Impostor samples (ONLY for test)
    idx_i = np.where(y != subject_id)[0]

    Xg = X[idx_g]
    Xi = X[idx_i]

    # Split genuine only
    Xtr, _, Xva, _, Xte_g, _ = split_epochs_per_subject(
        Xg, np.zeros(len(Xg)), train=0.65, val=0.15
    )

    # Test set = genuine + impostor
    Xte = np.concatenate([Xte_g, Xi], axis=0)
    yte = np.concatenate([
        np.ones(len(Xte_g), dtype=np.int32),    # genuine
        -np.ones(len(Xi), dtype=np.int32)       # impostor
    ])

    return Xtr, Xva, Xte, yte



# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing CSV files like Subject_01_PC.csv and Subject_01_VR.csv")
    ap.add_argument("--out_dir", type=str, default="./out",
                    help="Output folder where .npz will be saved")
    ap.add_argument("--train", type=float, default=0.65)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--n_imposters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    print(f"Reading data from: {data_dir}")
    X_pc, y_pc, X_vr, y_vr, subject_map = combine_epochs_by_subject(data_dir)

    print("\nEpoch-level pooled datasets:")
    print("PC X,y:", X_pc.shape, y_pc.shape, "subjects:", len(np.unique(y_pc)))
    print("VR X,y:", X_vr.shape, y_vr.shape, "subjects:", len(np.unique(y_vr)))

    # Split per subject
    XtrVR, ytrVR, XvaVR, yvaVR, XteVR, yteVR = split_epochs_per_subject(
        X_vr, y_vr, train=args.train, val=args.val, seed=args.seed
    )
    XtrPC, ytrPC, XvaPC, yvaPC, XtePC, ytePC = split_epochs_per_subject(
        X_pc, y_pc, train=args.train, val=args.val, seed=args.seed
    )

    # Build genuine/impostor trials (use different seeds per split/condition for determinism)
    Xtr_vr, ytr_vr, cid_tr_vr = genuine_imposter_trials(XtrVR, ytrVR, n_imposters=args.n_imposters, seed=args.seed + 10)
    Xva_vr, yva_vr, cid_va_vr = genuine_imposter_trials(XvaVR, yvaVR, n_imposters=args.n_imposters, seed=args.seed + 20)
    Xte_vr, yte_vr, cid_te_vr = genuine_imposter_trials(XteVR, yteVR, n_imposters=args.n_imposters, seed=args.seed + 30)

    Xtr_pc, ytr_pc, cid_tr_pc = genuine_imposter_trials(XtrPC, ytrPC, n_imposters=args.n_imposters, seed=args.seed + 40)
    Xva_pc, yva_pc, cid_va_pc = genuine_imposter_trials(XvaPC, yvaPC, n_imposters=args.n_imposters, seed=args.seed + 50)
    Xte_pc, yte_pc, cid_te_pc = genuine_imposter_trials(XtePC, ytePC, n_imposters=args.n_imposters, seed=args.seed + 60)

    # Convert to REVE format [B, C, T]
    Xtr_vr = to_reve_format(Xtr_vr)
    Xva_vr = to_reve_format(Xva_vr)
    Xte_vr = to_reve_format(Xte_vr)

    Xtr_pc = to_reve_format(Xtr_pc)
    Xva_pc = to_reve_format(Xva_pc)
    Xte_pc = to_reve_format(Xte_pc)

    print("\nVerification datasets (REVE format):")
    print("VR train:", Xtr_vr.shape, ytr_vr.shape, cid_tr_vr.shape)
    print("VR  val :", Xva_vr.shape, yva_vr.shape, cid_va_vr.shape)
    print("VR test :", Xte_vr.shape, yte_vr.shape, cid_te_vr.shape)

    print("PC train:", Xtr_pc.shape, ytr_pc.shape, cid_tr_pc.shape)
    print("PC  val :", Xva_pc.shape, yva_pc.shape, cid_va_pc.shape)
    print("PC test :", Xte_pc.shape, yte_pc.shape, cid_te_pc.shape)

    # Save
    out_path = out_dir / "pc_vr_verification_dataset_lphp10_50.npz"
    meta = {
        "fs": FS,
        "epoch_window_sec": [-(PRE_SAMPLES / FS), (POST_SAMPLES / FS)],
        "notch_freq": NOTCH_FREQ,
        "band": [BAND_LOW, BAND_HIGH],
        "filter_order": FILTER_ORDER,
        "split": {"train": args.train, "val": args.val, "test": 1.0 - args.train - args.val},
        "n_imposters": args.n_imposters,
        "seed": args.seed,
        "format": "X*: [B, C, T], y*: [B], cid*: [B]",
        "subject_map": subject_map,
        "ch_names": CHANNELS,
    }

    np.savez_compressed(
        out_path,
        # VR
        Xtr_vr=Xtr_vr, ytr_vr=ytr_vr, cid_tr_vr=cid_tr_vr,
        Xva_vr=Xva_vr, yva_vr=yva_vr, cid_va_vr=cid_va_vr,
        Xte_vr=Xte_vr, yte_vr=yte_vr, cid_te_vr=cid_te_vr,
        # PC
        Xtr_pc=Xtr_pc, ytr_pc=ytr_pc, cid_tr_pc=cid_tr_pc,
        Xva_pc=Xva_pc, yva_pc=yva_pc, cid_va_pc=cid_va_pc,
        Xte_pc=Xte_pc, yte_pc=yte_pc, cid_te_pc=cid_te_pc,
        # Metadata
        meta_json=json.dumps(meta),
    )

    print(f"\nSaved dataset to: {out_path}")
    print("You can load it with:")
    print(f"  data = np.load(r\"{out_path}\", allow_pickle=True)")
    print("  meta = json.loads(data['meta_json'].item())")

    print("\n==============================")
    print("Building ONE-CLASS datasets...")
    print("==============================")

    oneclass_data_vr = {}

    for subj in np.unique(y_vr):
        Xtr, Xva, Xte, yte = build_oneclass_splits(X_vr, y_vr, subj)

        oneclass_data_vr[int(subj)] = {
            "Xtr": to_reve_format(Xtr),
            "Xva": to_reve_format(Xva),
            "Xte": to_reve_format(Xte),
            "yte": yte.astype(np.int32)
        }

        print(
            f"[VR | One-Class] subj={subj} | "
            f"train={Xtr.shape[0]}, val={Xva.shape[0]}, test={Xte.shape[0]} "
            f"(genuine={np.sum(yte==1)}, impostor={np.sum(yte==-1)})"
        )

    oneclass_data_pc = {}

    for subj in np.unique(y_pc):
        Xtr, Xva, Xte, yte = build_oneclass_splits(X_pc, y_pc, subj)

        oneclass_data_pc[int(subj)] = {
            "Xtr": to_reve_format(Xtr),
            "Xva": to_reve_format(Xva),
            "Xte": to_reve_format(Xte),
            "yte": yte.astype(np.int32)
        }

        print(
            f"[PC | One-Class] subj={subj} | "
            f"train={Xtr.shape[0]}, val={Xva.shape[0]}, test={Xte.shape[0]} "
            f"(genuine={np.sum(yte==1)}, impostor={np.sum(yte==-1)})"
        )

    oneclass_out_path = out_dir / "pc_vr_oneclass_dataset.npz"

    np.savez_compressed(
        oneclass_out_path,
        vr=oneclass_data_vr,
        pc=oneclass_data_pc,
        meta_json=json.dumps(meta),
    )

    print(f"\nSaved ONE-CLASS dataset to: {oneclass_out_path}")






if __name__ == "__main__":
    main()

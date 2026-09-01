#!/usr/bin/env python3
"""Run supervised 5-fold PC/VR verification experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel

from cv5_metrics import compute_eer, metrics_at_threshold, summarize_metric_rows, threshold_at_validation_eer
from cv5_reve_cache import embed_with_cache
from feature_extractors import (
    build_eegnet_verifier,
    collect_scores_labels_eegnet,
    to_onehot,
    train_eegnet_end2end,
)
from head_utils import VerifierHead, train_head
from verify_utils import collect_scores_labels_head


SCENARIOS = {
    "PC->PC": ("pc", "pc", "pc"),
    "VR->VR": ("vr", "vr", "vr"),
    "PC->VR": ("pc", "pc", "vr"),
    "VR->PC": ("vr", "vr", "pc"),
    "Mixed->PC": ("mixed", "mixed", "pc"),
    "Mixed->VR": ("mixed", "mixed", "vr"),
}
REVE_MODELS = ["REVE+MLP", "REVE+SVM", "REVE+KNN", "REVE+RF"]
ALL_MODELS = REVE_MODELS + ["EEGNet"]
METRIC_COLS = ["accuracy", "balanced_accuracy", "FAR", "FRR", "EER"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to cv5_verification_dataset_lphp10_50.npz")
    ap.add_argument("--model_dir", default="./models", help="Folder containing reve-base and reve-positions")
    ap.add_argument("--out_dir", default="./out/cv5/results")
    ap.add_argument("--model", choices=ALL_MODELS + ["all"], required=True)
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    ap.add_argument("--use_claimed_id", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reve_bs", type=int, default=64)
    ap.add_argument("--head_bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache_dir", default="./out/cv5/cache/reve")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_reve(model_dir, device):
    model = AutoModel.from_pretrained(
        str(Path(model_dir) / "reve-base"),
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
    ).eval().to(device)
    pos_bank = AutoModel.from_pretrained(
        str(Path(model_dir) / "reve-positions"),
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
    ).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in pos_bank.parameters():
        p.requires_grad = False
    return model, pos_bank


def split_arrays(data, env, fold, split):
    prefix = f"{env}_fold{fold}_{split}"
    return (
        data[f"X_{prefix}"].astype(np.float32),
        data[f"y_{prefix}"].astype(np.int32),
        data[f"cid_{prefix}"].astype(np.int64),
    )


def scenario_arrays(data, scenario, fold):
    train_env, val_env, test_env = SCENARIOS[scenario]
    if train_env == "mixed":
        Xtr_pc, ytr_pc, cid_tr_pc = split_arrays(data, "pc", fold, "train")
        Xtr_vr, ytr_vr, cid_tr_vr = split_arrays(data, "vr", fold, "train")
        Xva_pc, yva_pc, cid_va_pc = split_arrays(data, "pc", fold, "val")
        Xva_vr, yva_vr, cid_va_vr = split_arrays(data, "vr", fold, "val")
        Xtr = np.concatenate([Xtr_pc, Xtr_vr], axis=0)
        ytr = np.concatenate([ytr_pc, ytr_vr], axis=0)
        cid_tr = np.concatenate([cid_tr_pc, cid_tr_vr], axis=0)
        Xva = np.concatenate([Xva_pc, Xva_vr], axis=0)
        yva = np.concatenate([yva_pc, yva_vr], axis=0)
        cid_va = np.concatenate([cid_va_pc, cid_va_vr], axis=0)
    else:
        Xtr, ytr, cid_tr = split_arrays(data, train_env, fold, "train")
        Xva, yva, cid_va = split_arrays(data, val_env, fold, "val")
    Xte, yte, cid_te = split_arrays(data, test_env, fold, "test")
    return Xtr, ytr, cid_tr, Xva, yva, cid_va, Xte, yte, cid_te


def add_claimed_id(Z, cid, n_subjects):
    return np.concatenate([Z, np.eye(n_subjects, dtype=np.float32)[cid.astype(int)]], axis=1)


def make_sklearn(model_name, seed):
    if model_name == "REVE+SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed))
    if model_name == "REVE+KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=4))
    if model_name == "REVE+RF":
        return RandomForestClassifier(n_estimators=300, random_state=seed)
    raise ValueError(model_name)


def eval_from_scores(model_name, scenario, fold, val_scores, yva, test_scores, yte, seed):
    threshold = threshold_at_validation_eer(val_scores, yva)
    row = {
        "model": model_name,
        "scenario": scenario,
        "fold": fold,
        "threshold": float(threshold),
        "seed": seed,
        "EER": compute_eer(test_scores, yte),
    }
    row.update(metrics_at_threshold(test_scores, yte, threshold))
    return row


def run_reve_model(args, data, meta, device, model_name, scenario, fold, reve_model, pos_bank):
    Xtr, ytr, cid_tr, Xva, yva, cid_va, Xte, yte, cid_te = scenario_arrays(data, scenario, fold)
    ch_names = meta["ch_names"]
    n_subjects = len(meta["subject_map"])
    cache = not args.no_cache

    Ztr = embed_with_cache(reve_model, pos_bank, Xtr, ch_names, device, args.reve_bs, args.cache_dir, args.model_dir, f"{scenario}_fold{fold}_train", cache)
    Zva = embed_with_cache(reve_model, pos_bank, Xva, ch_names, device, args.reve_bs, args.cache_dir, args.model_dir, f"{scenario}_fold{fold}_val", cache)
    Zte = embed_with_cache(reve_model, pos_bank, Xte, ch_names, device, args.reve_bs, args.cache_dir, args.model_dir, f"{scenario}_fold{fold}_test", cache)

    if model_name in ["REVE+SVM", "REVE+KNN", "REVE+RF"]:
        if args.use_claimed_id:
            Ztr = add_claimed_id(Ztr, cid_tr, n_subjects)
            Zva = add_claimed_id(Zva, cid_va, n_subjects)
            Zte = add_claimed_id(Zte, cid_te, n_subjects)
        clf = make_sklearn(model_name, args.seed + fold)
        clf.fit(Ztr, ytr)
        val_scores = clf.predict_proba(Zva)[:, 1]
        test_scores = clf.predict_proba(Zte)[:, 1]
        return eval_from_scores(model_name, scenario, fold, val_scores, yva, test_scores, yte, args.seed)

    train_ds = TensorDataset(
        torch.tensor(Ztr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32),
        torch.tensor(cid_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(Zva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.float32),
        torch.tensor(cid_va, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(Zte, dtype=torch.float32),
        torch.tensor(yte, dtype=torch.float32),
        torch.tensor(cid_te, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.head_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.head_bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.head_bs, shuffle=False)
    head = VerifierHead(Ztr.shape[1], args.use_claimed_id, n_subjects).to(device)
    train_head(head, train_loader, val_loader, device, args.use_claimed_id, args.lr, args.epochs)
    val_scores, val_labels = collect_scores_labels_head(head, val_loader, device, args.use_claimed_id)
    test_scores, test_labels = collect_scores_labels_head(head, test_loader, device, args.use_claimed_id)
    return eval_from_scores(model_name, scenario, fold, val_scores, val_labels, test_scores, test_labels, args.seed)


def eegnet_loader(X, y, cid, use_claimed_id, n_subjects, batch_size):
    if use_claimed_id:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(to_onehot(cid, n_subjects), dtype=torch.float32),
        )
    else:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_eegnet(args, data, meta, device, scenario, fold):
    Xtr, ytr, cid_tr, Xva, yva, cid_va, Xte, yte, cid_te = scenario_arrays(data, scenario, fold)
    n_subjects = len(meta["subject_map"])
    model = build_eegnet_verifier(16, Xtr.shape[-1], n_subjects, device, args.use_claimed_id)
    _, val_threshold = train_eegnet_end2end(
        model,
        Xtr,
        ytr,
        cid_tr,
        Xva,
        yva,
        cid_va,
        device=device,
        n_subjects=n_subjects,
        use_claimed_id=args.use_claimed_id,
        lr=args.lr,
        epochs=args.epochs,
        bs=args.head_bs,
    )
    val_loader = eegnet_loader(Xva, yva, cid_va, args.use_claimed_id, n_subjects, args.head_bs)
    test_loader = eegnet_loader(Xte, yte, cid_te, args.use_claimed_id, n_subjects, args.head_bs)
    val_scores, val_labels = collect_scores_labels_eegnet(model, val_loader, device, args.use_claimed_id)
    test_scores, test_labels = collect_scores_labels_eegnet(model, test_loader, device, args.use_claimed_id)
    row = eval_from_scores("EEGNet", scenario, fold, val_scores, val_labels, test_scores, test_labels, args.seed)
    row["threshold"] = float(val_threshold)
    row.update(metrics_at_threshold(test_scores, test_labels, val_threshold))
    return row


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = np.load(args.npz, allow_pickle=True)
    meta = json.loads(data["meta_json"].item())

    models = ALL_MODELS if args.model == "all" else [args.model]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    needs_reve = any(m.startswith("REVE+") for m in models)
    reve_model, pos_bank = load_reve(args.model_dir, device) if needs_reve else (None, None)

    rows = []
    for model_name in models:
        for scenario in scenarios:
            for fold in range(int(meta["n_folds"])):
                set_seed(args.seed + fold)
                print(f"[RUN] {model_name} | {scenario} | fold {fold}")
                if model_name == "EEGNet":
                    row = run_eegnet(args, data, meta, device, scenario, fold)
                else:
                    row = run_reve_model(args, data, meta, device, model_name, scenario, fold, reve_model, pos_bank)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "supervised_cv5_per_fold.csv", index=False)

    summary = summarize_metric_rows(rows, ["model", "scenario", "seed"], METRIC_COLS)
    summary.to_csv(out_dir / "supervised_cv5_summary.csv", index=False)
    print(f"Saved {out_dir / 'supervised_cv5_per_fold.csv'}")
    print(f"Saved {out_dir / 'supervised_cv5_summary.csv'}")


if __name__ == "__main__":
    main()

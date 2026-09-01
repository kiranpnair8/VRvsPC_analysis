#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import json

from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors

from feature_extractors import (
    load_reve_local,
    reve_embed_batch,
    compute_eer,
    far_frr_at_threshold
)




# -------------------------------------------------
# Args
# -------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True,
                    help="Path to pc_vr_oneclass_dataset.npz")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Directory containing reve-base and reve-positions")
    ap.add_argument("--env", type=str, choices=["pc", "vr", "mixed"], required=True,
                    help="Training environment")
    ap.add_argument("--test_env", type=str, choices=["pc", "vr"], required=True,
                    help="Testing environment")
    ap.add_argument("--model", type=str, choices=["ocsvm", "ocknn"], required=True,
                    help="One-class model type")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bs", type=int, default=128,
                    help="Batch size for REVE embedding extraction")
    return ap.parse_args()


# -------------------------------------------------
# REVE embedding helper
# -------------------------------------------------
def extract_reve_embeddings(model, pos_bank, X, device, ch_names, bs):
    Z = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        zb = reve_embed_batch(model, pos_bank, xb, ch_names, device)
        Z.append(zb.detach().cpu().numpy())
    return np.vstack(Z)


# -------------------------------------------------
# One-class models
# -------------------------------------------------
def train_ocsvm(Ztr, nu=0.1):
    clf = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    clf.fit(Ztr)
    return clf


def train_ocknn(Ztr, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Ztr)
    return nn


def score_ocsvm(clf, Z):
    return clf.decision_function(Z)   # higher = more genuine


def score_ocknn(nn, Z):
    dists, _ = nn.kneighbors(Z)
    return -np.mean(dists, axis=1)    # higher = more genuine


# -------------------------------------------------
# Per-subject evaluation
# -------------------------------------------------
def eval_one_subject(Ztr, Zva, Zte, yte, model_type):
    if model_type == "ocsvm":
        clf = train_ocsvm(Ztr)
        val_scores = score_ocsvm(clf, Zva)
        test_scores = score_ocsvm(clf, Zte)

    elif model_type == "ocknn":
        nn = train_ocknn(Ztr)
        val_scores = score_ocknn(nn, Zva)
        test_scores = score_ocknn(nn, Zte)

    else:
        raise ValueError("Unknown model type")

    # Validation threshold
    val_eer, val_thr = compute_eer(
        val_scores, np.ones(len(val_scores)), return_thr=True
    )

    # Test metrics
    test_eer = compute_eer(test_scores, yte)
    test_far, test_frr = far_frr_at_threshold(test_scores, yte, val_thr)

    return val_eer, test_eer, test_far, test_frr


# -------------------------------------------------
# Experiment runner
# -------------------------------------------------
def run_experiment(
    pc_data,
    vr_data,
    reve_model,
    pos_bank,
    device,
    ch_names,
    train_env,
    test_env,
    model_type,
    bs,
):
    assert not (train_env == "mixed" and test_env == "mixed"), \
        "Mixed → Mixed testing is undefined for one-class verification."

    all_val_eer, all_eer, all_far, all_frr = [], [], [], []

    for subj in pc_data.keys():

        # ---- training data ----
        if train_env == "pc":
            Xtr, Xva = pc_data[subj]["Xtr"], pc_data[subj]["Xva"]

        elif train_env == "vr":
            Xtr, Xva = vr_data[subj]["Xtr"], vr_data[subj]["Xva"]

        else:  # mixed
            Xtr = np.concatenate([pc_data[subj]["Xtr"], vr_data[subj]["Xtr"]], axis=0)
            Xva = np.concatenate([pc_data[subj]["Xva"], vr_data[subj]["Xva"]], axis=0)

        # ---- test data ----
        if test_env == "pc":
            Xte, yte = pc_data[subj]["Xte"], pc_data[subj]["yte"]
        else:
            Xte, yte = vr_data[subj]["Xte"], vr_data[subj]["yte"]



        # ---- embeddings ----
        Ztr = extract_reve_embeddings(reve_model, pos_bank, Xtr, device, ch_names, bs)
        Zva = extract_reve_embeddings(reve_model, pos_bank, Xva, device, ch_names, bs)
        Zte = extract_reve_embeddings(reve_model, pos_bank, Xte, device, ch_names, bs)

        val_eer, test_eer, far, frr = eval_one_subject(
            Ztr, Zva, Zte, yte, model_type
        )

        all_val_eer.append(val_eer)
        all_eer.append(test_eer)
        all_far.append(far)
        all_frr.append(frr)

    return (
        float(np.mean(all_val_eer)),
        float(np.mean(all_eer)),
        float(np.mean(all_far)),
        float(np.mean(all_frr)),
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = get_args()
    device = torch.device(args.device)

    pc_data, vr_data, meta = load_oneclass_data(args.npz)

    if "ch_names" not in meta:
        raise RuntimeError("meta_json must contain 'ch_names' for REVE embeddings")

    ch_names = meta["ch_names"]

    # ---- REVE ----
    reve_model, pos_bank = load_reve_local(args.model_dir, device)
    reve_model.eval()
    reve_model.to(device)

    # ---- channel names (must exist) ----
    if "ch_names" not in meta:
        raise RuntimeError("meta_json must contain 'ch_names' for REVE embeddings")
    ch_names = meta["ch_names"]

    val_eer, eer, far, frr = run_experiment(
        pc_data,
        vr_data,
        reve_model,
        pos_bank,
        device,
        ch_names,
        train_env=args.env,
        test_env=args.test_env,
        model_type=args.model,
        bs=args.bs,
    )

    print("\n==============================")
    print("ONE-CLASS VERIFICATION RESULT")
    print("==============================")
    print(f"MODEL      : {args.model}")
    print(f"TRAIN ENV  : {args.env}")
    print(f"TEST ENV   : {args.test_env}")
    print(f"VAL EER    : {val_eer*100:.2f}%")
    print(f"EER        : {eer*100:.2f}%")
    print(f"FAR@ValThr : {far*100:.2f}%")
    print(f"FRR@ValThr : {frr*100:.2f}%")


def load_oneclass_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    pc = data["pc"].item()
    vr = data["vr"].item()
    meta = json.loads(data["meta_json"].item())
    return pc, vr, meta


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train verifier (REVE frozen + head OR EEGNet end-to-end) and evaluate with:
- Val EER
- Test EER
- Test FAR@ValThr
- Test FRR@ValThr

Run:
python train_reve_verifier_eer.py --npz "out/pc_vr_verification_dataset.npz" --model_dir "./models" --use_claimed_id
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from transformers import AutoModel
from braindecode.models import EEGNet

from feature_extractors import (
    load_reve_local,
    get_positions,
    reve_embed_batch,
    to_onehot,
    compute_eer,
    far_frr_at_threshold,
    build_eegnet_verifier,
    collect_scores_labels_eegnet,
    train_eegnet_end2end,
    finetune_eegnet
)
from mixed_run import (
    merge_pc_vr_splits,
    run_mixed_train_eval
)
from head_utils import (
    train_sklearn_model,
    tune_and_train_sklearn
)

# -----------------------
# Config
# -----------------------
CHANNELS = ["Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"]

MODEL = "RF"  # "MLP", "SVM", "KNN", "RF" (only for REVE path)
FEATURE_EXTRACTOR = "EEGNET"  # "REVE" or "EEGNET"

PARAM_GRIDS = {
    "SVM": {
        "C": [1, 10],
        "gamma": ["scale"]
    },
    "KNN": {
        "n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9]
    },
    "RF": {
        "n_estimators": [100, 150, 200, 250, 300, 350]
    }
}

#eegnet fine-tuning parms
LR_LIST = [1e-3, 5e-4, 1e-4]
EPOCH_LIST = [20, 25, 30, 35]



class VerifierHead(nn.Module):
    def __init__(self, emb_dim: int, use_claimed_id: bool, n_subjects: int, id_dim: int = 32):
        super().__init__()
        self.use_claimed_id = use_claimed_id
        self.id_emb = None
        in_dim = emb_dim

        if use_claimed_id:
            self.id_emb = nn.Embedding(n_subjects, id_dim)
            in_dim += id_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z: torch.Tensor, cid: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_claimed_id:
            assert cid is not None
            z = torch.cat([z, self.id_emb(cid)], dim=1)
        return self.net(z).squeeze(1)


def make_embedding_dataset(model, pos_bank, X, y, cid, device, batch_size, ch_names):
    mask = np.isfinite(X).all(axis=(1, 2))
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[WARN] Dropping {dropped}/{len(X)} epochs due to NaN/Inf")

    X = X[mask]
    y = y[mask]
    cid = cid[mask]

    zs, ys, cids = [], [], []
    n = X.shape[0]

    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        zb = reve_embed_batch(model, pos_bank, xb, ch_names, device)
        zs.append(zb.detach().cpu())
        ys.append(torch.tensor(y[i:i+batch_size], dtype=torch.float32))
        cids.append(torch.tensor(cid[i:i+batch_size], dtype=torch.long))

    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0)
    CID = torch.cat(cids, dim=0)
    return TensorDataset(Z, Y, CID)


def train_head(head: nn.Module, train_loader, val_loader, device: torch.device,
               use_claimed_id: bool, lr: float, epochs: int):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_eer = 1.0
    best_state = None

    for ep in range(1, epochs + 1):
        head.train()
        losses = []

        for z, y, cid in train_loader:
            z = z.to(device)
            y = y.to(device)
            cid = cid.to(device)

            opt.zero_grad()
            logits = head(z, cid if use_claimed_id else None)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # val EER
        val_scores, val_labels = collect_scores_labels_head(head, val_loader, device, use_claimed_id)
        val_eer = compute_eer(val_scores, val_labels)

        print(f"Epoch {ep:02d} | train loss {np.mean(losses):.4f} | val EER {val_eer*100:.2f}%")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    return best_val_eer


@torch.no_grad()
def collect_scores_labels_head(head: nn.Module, loader, device: torch.device, use_claimed_id: bool):
    head.eval()
    all_scores, all_labels = [], []

    for z, y, cid in loader:
        z = z.to(device)
        cid = cid.to(device)

        logits = head(z, cid if use_claimed_id else None).view(-1)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

        all_scores.append(scores)
        all_labels.append(y.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)




def add_cid_onehot(Z: np.ndarray, cid: np.ndarray, n_subjects: int) -> np.ndarray:
    cid = cid.astype(int).ravel()
    onehot = np.eye(n_subjects, dtype=np.float32)[cid]
    return np.concatenate([Z, onehot], axis=1)


def sklearn_scores(clf, Z):
    return clf.predict_proba(Z)[:, 1]


def eval_sklearn_valthr_metrics(clf, Zva, yva, Zte, yte):
    val_scores = sklearn_scores(clf, Zva)
    val_eer, val_thr = compute_eer(val_scores, yva, return_thr=True)

    test_scores = sklearn_scores(clf, Zte)
    test_eer = compute_eer(test_scores, yte)

    test_far, test_frr = far_frr_at_threshold(test_scores, yte, val_thr)
    return float(val_eer), float(test_eer), float(test_far), float(test_frr)


# -----------------------
# EEGNet end-to-end
# -----------------------



class EEGNetClaimedIDVerifier(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_subjects: int, device):
        super().__init__()

        self.backbone = EEGNet(
            n_chans=n_chans,
            n_outputs=32,
            n_times=n_times,
            final_conv_length="auto"
        ).to(device)

        self.head = nn.Sequential(
            nn.Linear(32 + n_subjects, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, x_bct, cid_onehot):
        feats = self.backbone(x_bct)
        z = torch.cat([feats, cid_onehot], dim=1)
        return self.head(z).squeeze(1)



# -----------------------
# Runner
# -----------------------
def run_condition(train_tag: str, test_tag: str, data,
                  model, pos_bank, device, args, n_subjects: int):

    Xtr = data[f"Xtr_{train_tag}"].astype(np.float32)
    ytr = data[f"ytr_{train_tag}"].astype(np.float32)
    cid_tr = data[f"cid_tr_{train_tag}"].astype(np.int64)

    Xva = data[f"Xva_{train_tag}"].astype(np.float32)
    yva = data[f"yva_{train_tag}"].astype(np.float32)
    cid_va = data[f"cid_va_{train_tag}"].astype(np.int64)

    Xte = data[f"Xte_{test_tag}"].astype(np.float32)
    yte = data[f"yte_{test_tag}"].astype(np.float32)
    cid_te = data[f"cid_te_{test_tag}"].astype(np.int64)

    print("Loaded shapes:")
    print("  train:", Xtr.shape, ytr.shape, cid_tr.shape)
    print("  val  :", Xva.shape, yva.shape, cid_va.shape)
    print("  test :", Xte.shape, yte.shape, cid_te.shape)

    # -------------------------
    # EEGNET end-to-end
    # -------------------------
    if FEATURE_EXTRACTOR == "EEGNET":
        print("Training EEGNet end-to-end ...")

        # drop bad epochs
        mask_tr = np.isfinite(Xtr).all(axis=(1, 2))
        mask_va = np.isfinite(Xva).all(axis=(1, 2))
        mask_te = np.isfinite(Xte).all(axis=(1, 2))

        Xtr2, ytr2, cid_tr2 = Xtr[mask_tr], ytr[mask_tr], cid_tr[mask_tr]
        Xva2, yva2, cid_va2 = Xva[mask_va], yva[mask_va], cid_va[mask_va]
        Xte2, yte2, cid_te2 = Xte[mask_te], yte[mask_te], cid_te[mask_te]

        use_cid = bool(args.use_claimed_id)

        if args.tune_eegnet:
            def build_fn():
                return build_eegnet_verifier(
                    n_chans=16,
                    n_times=Xtr2.shape[-1],
                    n_subjects=n_subjects,
                    device=device,
                    use_claimed_id=use_cid
                )

            eegnet, best_lr, best_epochs, best_val_eer, best_val_thr = finetune_eegnet(
                build_eegnet_fn=build_fn,
                Xtr=Xtr2, ytr=ytr2, cid_tr=cid_tr2 if use_cid else None,
                Xva=Xva2, yva=yva2, cid_va=cid_va2 if use_cid else None,
                device=device,
                n_subjects=n_subjects,
                use_claimed_id=use_cid,
                LR_LIST=LR_LIST,
                EPOCH_LIST=EPOCH_LIST,
                bs=args.head_bs,
            )

        else:
            eegnet = build_eegnet_verifier(
                n_chans=16,
                n_times=Xtr2.shape[-1],
                n_subjects=n_subjects,
                device=device,
                use_claimed_id=use_cid
            )

            best_val_eer, best_val_thr = train_eegnet_end2end(
                eegnet,
                Xtr2, ytr2, cid_tr2 if use_cid else None,
                Xva2, yva2, cid_va2 if use_cid else None,
                device=device, n_subjects=n_subjects,
                use_claimed_id=use_cid,
                lr=args.lr, epochs=args.epochs, bs=args.head_bs
            )


        # test loader
        if use_cid:
            cid_te_oh = torch.tensor(to_onehot(cid_te2, n_subjects), dtype=torch.float32)
            test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xte2, dtype=torch.float32),
                    torch.tensor(yte2, dtype=torch.float32),
                    cid_te_oh
                ),
                batch_size=args.head_bs, shuffle=False
            )
        else:
            test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xte2, dtype=torch.float32),
                    torch.tensor(yte2, dtype=torch.float32)
                ),
                batch_size=args.head_bs, shuffle=False
            )

        test_scores, test_labels = collect_scores_labels_eegnet(eegnet, test_loader, device, use_cid)
        test_eer = compute_eer(test_scores, test_labels)
        test_far, test_frr = far_frr_at_threshold(test_scores, test_labels, best_val_thr)

        print(
            f"[{train_tag.upper()}→{test_tag.upper()}] EEGNET "
            f"Val EER: {best_val_eer*100:.2f}% | "
            f"Test EER: {test_eer*100:.2f}% | "
            f"Test FAR@ValThr: {test_far*100:.2f}% | "
            f"Test FRR@ValThr: {test_frr*100:.2f}%"
        )

        return float(best_val_eer), float(test_eer), float(test_far), float(test_frr)

    # -------------------------
    # REVE frozen + head
    # -------------------------
    print("Extracting REVE embeddings (frozen) ...")

    ds_tr = make_embedding_dataset(model, pos_bank, Xtr, ytr, cid_tr, device, args.reve_bs, CHANNELS)
    ds_va = make_embedding_dataset(model, pos_bank, Xva, yva, cid_va, device, args.reve_bs, CHANNELS)
    ds_te = make_embedding_dataset(model, pos_bank, Xte, yte, cid_te, device, args.reve_bs, CHANNELS)

    # sklearn path
    if MODEL in ["SVM", "KNN", "RF"]:
        Ztr = ds_tr.tensors[0].numpy()
        ytr_np = ds_tr.tensors[1].numpy().astype(int)
        cid_tr_np = ds_tr.tensors[2].numpy()

        Zva = ds_va.tensors[0].numpy()
        yva_np = ds_va.tensors[1].numpy().astype(int)
        cid_va_np = ds_va.tensors[2].numpy()

        Zte = ds_te.tensors[0].numpy()
        yte_np = ds_te.tensors[1].numpy().astype(int)
        cid_te_np = ds_te.tensors[2].numpy()

        if args.use_claimed_id:
            Ztr = add_cid_onehot(Ztr, cid_tr_np, n_subjects)
            Zva = add_cid_onehot(Zva, cid_va_np, n_subjects)
            Zte = add_cid_onehot(Zte, cid_te_np, n_subjects)

        if args.tune:
            clf, best_params, best_val_eer = tune_and_train_sklearn(
                model_name=MODEL,
                Ztr=Ztr,
                ytr=ytr_np,
                Zval=Zva,
                yval=yva_np,
                param_grid=PARAM_GRIDS[MODEL]
            )
        else:
            clf = train_sklearn_model(MODEL, Ztr, ytr_np)

        val_eer, test_eer, test_far, test_frr = eval_sklearn_valthr_metrics(clf, Zva, yva_np, Zte, yte_np)

        print(
            f"[{train_tag.upper()}→{test_tag.upper()}] REVE+{MODEL} "
            f"Val EER: {val_eer*100:.2f}% | "
            f"Test EER: {test_eer*100:.2f}% | "
            f"Test FAR@ValThr: {test_far*100:.2f}% | "
            f"Test FRR@ValThr: {test_frr*100:.2f}%"
        )

        return float(val_eer), float(test_eer), float(test_far), float(test_frr)

    # MLP head path
    train_loader = DataLoader(ds_tr, batch_size=args.head_bs, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=args.head_bs, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=args.head_bs, shuffle=False)

    z0, _, _ = ds_tr[0]
    emb_dim = int(z0.shape[0])

    head = VerifierHead(emb_dim=emb_dim, use_claimed_id=args.use_claimed_id, n_subjects=n_subjects).to(device)
    print(f"Training REVE+MLP head (use_claimed_id={args.use_claimed_id}) ...")

    best_val_eer = train_head(head, train_loader, val_loader, device, args.use_claimed_id, args.lr, args.epochs)

    # VAL threshold
    val_scores, val_labels = collect_scores_labels_head(head, val_loader, device, args.use_claimed_id)
    val_eer, val_thr = compute_eer(val_scores, val_labels, return_thr=True)

    # TEST metrics at VAL threshold
    test_scores, test_labels = collect_scores_labels_head(head, test_loader, device, args.use_claimed_id)
    test_eer = compute_eer(test_scores, test_labels)
    test_far, test_frr = far_frr_at_threshold(test_scores, test_labels, val_thr)

    print(
        f"[{train_tag.upper()}→{test_tag.upper()}] REVE+MLP "
        f"Val EER: {val_eer*100:.2f}% | "
        f"Test EER: {test_eer*100:.2f}% | "
        f"Test FAR@ValThr: {test_far*100:.2f}% | "
        f"Test FRR@ValThr: {test_frr*100:.2f}%"
    )

    return float(val_eer), float(test_eer), float(test_far), float(test_frr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross_env", action="store_true",
                    help="Enable cross-environment evaluation (PC→VR, VR→PC)")
    ap.add_argument("--train_mixed", action="store_true",
                    help="Train on PC+VR together, then test separately on PC and VR")
    ap.add_argument("--model_name", type=str, default="MLP", choices=["MLP", "SVM", "KNN", "RF"],
                    help="Classifier head model (REVE path only)")
    ap.add_argument("--npz", type=str, required=True, help="Path to pc_vr_verification_dataset.npz")
    ap.add_argument("--model_dir", type=str, required=True, help="Folder containing reve-base and reve-positions")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_claimed_id", action="store_true", help="Condition head on claimed subject id")
    ap.add_argument("--reve_bs", type=int, default=64, help="Batch size for REVE embedding extraction")
    ap.add_argument("--head_bs", type=int, default=256, help="Batch size for head training")
    ap.add_argument("--tune", action="store_true", help="Validation-based hyperparameter tuning")
    ap.add_argument(
    "--tune_eegnet",
    action="store_true",
    help="Validation-based fine-tuning of EEGNet (lr × epochs)"
)


    args = ap.parse_args()

    CROSS_ENV = args.cross_env

    # print mode
    if args.train_mixed:
        mode_str = "MIXED-TRAIN (PC+VR) → TEST (PC, VR)"
    else:
        mode_str = "CROSS-ENV (PC↔VR)" if CROSS_ENV else "WITHIN-ENV (PC→PC, VR→VR)"

    print(f"Evaluation mode: {mode_str}")
    print(f"MODEL: {MODEL} | FE={FEATURE_EXTRACTOR} | use_claimed_id={args.use_claimed_id} | cross_env={args.cross_env}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = np.load(args.npz, allow_pickle=True)
    meta = json.loads(data["meta_json"].item())
    n_subjects = len(meta["subject_map"])
    print("Subjects:", n_subjects)
    print("n_imposters:", meta["n_imposters"], "| split:", meta["split"])

    # Load feature extractor (REVE only if needed)
    model, pos_bank = None, None
    eegnet = None
    if FEATURE_EXTRACTOR == "REVE":
        model, pos_bank = load_reve_local(args.model_dir, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    # NEW MIXED MODE
    if args.train_mixed:
        run_mixed_train_eval(data, model, pos_bank, eegnet, device, args, n_subjects)
        return

    # -----------------------------
    # OLD within-env / cross-env
    # -----------------------------
    if not CROSS_ENV:
        pc_val, pc_test, pc_far, pc_frr = run_condition(
            "pc", "pc", data, model, pos_bank, device, args, n_subjects
        )
        vr_val, vr_test, vr_far, vr_frr = run_condition(
            "vr", "vr", data, model, pos_bank, device, args, n_subjects
        )
    else:
        pc_val, pc_test, pc_far, pc_frr = run_condition(
            "pc", "vr", data, model, pos_bank, device, args, n_subjects
        )
        vr_val, vr_test, vr_far, vr_frr = run_condition(
            "vr", "pc", data, model, pos_bank, device, args, n_subjects
        )

    print("\n==================== SUMMARY ====================")
    print(f"PC: val EER {pc_val*100:.2f}% | test EER {pc_test*100:.2f}%")
    print(f"VR: val EER {vr_val*100:.2f}% | test EER {vr_test*100:.2f}%")

    print(f"\nPC test FAR@ValThr: {pc_far*100:.2f}%")
    print(f"VR test FAR@ValThr: {vr_far*100:.2f}%")

    print(f"\nPC test FRR@ValThr: {pc_frr*100:.2f}%")
    print(f"VR test FRR@ValThr: {vr_frr*100:.2f}%")

    print("\nLower EER is better (0% ideal, 50% ~ random).")


if __name__ == "__main__":
    main()


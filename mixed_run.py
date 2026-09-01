# mixed_run.py
# mix env helper

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from verify_utils import (
    collect_scores_labels_head
)

from feature_extractors import (
    build_eegnet_verifier,
    train_eegnet_end2end,
    collect_scores_labels_eegnet,
    far_frr_at_threshold,
    compute_eer,
    to_onehot,
)
from head_utils import (
    VerifierHead,
    train_head,
    make_embedding_dataset,
    train_sklearn_model
)


CHANNELS = ["Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"]

def merge_pc_vr_splits(data):
    Xtr_pc = data["Xtr_pc"].astype(np.float32)
    ytr_pc = data["ytr_pc"].astype(np.float32)
    cid_tr_pc = data["cid_tr_pc"].astype(np.int64)

    Xva_pc = data["Xva_pc"].astype(np.float32)
    yva_pc = data["yva_pc"].astype(np.float32)
    cid_va_pc = data["cid_va_pc"].astype(np.int64)

    Xtr_vr = data["Xtr_vr"].astype(np.float32)
    ytr_vr = data["ytr_vr"].astype(np.float32)
    cid_tr_vr = data["cid_tr_vr"].astype(np.int64)

    Xva_vr = data["Xva_vr"].astype(np.float32)
    yva_vr = data["yva_vr"].astype(np.float32)
    cid_va_vr = data["cid_va_vr"].astype(np.int64)

    Xtr_mix = np.concatenate([Xtr_pc, Xtr_vr], axis=0)
    ytr_mix = np.concatenate([ytr_pc, ytr_vr], axis=0)
    cid_tr_mix = np.concatenate([cid_tr_pc, cid_tr_vr], axis=0)

    Xva_mix = np.concatenate([Xva_pc, Xva_vr], axis=0)
    yva_mix = np.concatenate([yva_pc, yva_vr], axis=0)
    cid_va_mix = np.concatenate([cid_va_pc, cid_va_vr], axis=0)

    return Xtr_mix, ytr_mix, cid_tr_mix, Xva_mix, yva_mix, cid_va_mix



def run_mixed_train_eval(data, model, pos_bank, eegnet, device, args, n_subjects):
    print("\n==================== MIXED TRAINING ====================")
    print("TRAIN: PC+VR")

    use_reve = (model is not None and pos_bank is not None)

    if use_reve:
        print("[MIXED] Feature Extractor = REVE")
    else:
        print("[MIXED] Feature Extractor = EEGNET")

    # ---- load mixed train+val ----
    Xtr, ytr, cid_tr, Xva, yva, cid_va = merge_pc_vr_splits(data)

    # ---- load test sets separately ----
    Xte_pc = data["Xte_pc"].astype(np.float32)
    yte_pc = data["yte_pc"].astype(np.float32)
    cid_te_pc = data["cid_te_pc"].astype(np.int64)

    Xte_vr = data["Xte_vr"].astype(np.float32)
    yte_vr = data["yte_vr"].astype(np.float32)
    cid_te_vr = data["cid_te_vr"].astype(np.int64)

    # ---- drop NaNs ----
    mask_tr = np.isfinite(Xtr).all(axis=(1, 2))
    mask_va = np.isfinite(Xva).all(axis=(1, 2))
    mask_pc = np.isfinite(Xte_pc).all(axis=(1, 2))
    mask_vr = np.isfinite(Xte_vr).all(axis=(1, 2))

    Xtr, ytr, cid_tr = Xtr[mask_tr], ytr[mask_tr], cid_tr[mask_tr]
    Xva, yva, cid_va = Xva[mask_va], yva[mask_va], cid_va[mask_va]
    Xte_pc, yte_pc, cid_te_pc = Xte_pc[mask_pc], yte_pc[mask_pc], cid_te_pc[mask_pc]
    Xte_vr, yte_vr, cid_te_vr = Xte_vr[mask_vr], yte_vr[mask_vr], cid_te_vr[mask_vr]

    use_cid = bool(args.use_claimed_id)

    # ======================================================
    # EEGNET FEATURE EXTRACTOR
    # ======================================================
    if not use_reve:
        print("Feature Extractor: EEGNET (end-to-end)")

        eegnet = build_eegnet_verifier(
            n_chans=16,
            n_times=Xtr.shape[-1],
            n_subjects=n_subjects,
            device=device,
            use_claimed_id=use_cid
        )

        best_val_eer = train_eegnet_end2end(
            eegnet,
            Xtr, ytr, cid_tr,
            Xva, yva, cid_va,
            device=device,
            n_subjects=n_subjects,
            use_claimed_id=use_cid,
            lr=args.lr,
            epochs=args.epochs,
            bs=args.head_bs
        )

        # ---- get val threshold ----
        if use_cid:
            cid_va_oh = torch.tensor(to_onehot(cid_va, n_subjects), dtype=torch.float32)
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xva, dtype=torch.float32),
                    torch.tensor(yva, dtype=torch.float32),
                    cid_va_oh
                ),
                batch_size=args.head_bs,
                shuffle=False
            )
        else:
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xva, dtype=torch.float32),
                    torch.tensor(yva, dtype=torch.float32)
                ),
                batch_size=args.head_bs,
                shuffle=False
            )

        val_scores, val_labels = collect_scores_labels_eegnet(eegnet, val_loader, device, use_claimed_id=use_cid)
        val_eer, val_thr = compute_eer(val_scores, val_labels, return_thr=True)

        # ---- test PC ----
        if use_cid:
            cid_pc_oh = torch.tensor(to_onehot(cid_te_pc, n_subjects), dtype=torch.float32)
            pc_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xte_pc, dtype=torch.float32),
                    torch.tensor(yte_pc, dtype=torch.float32),
                    cid_pc_oh
                ),
                batch_size=args.head_bs,
                shuffle=False
            )
        else:
            pc_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xte_pc, dtype=torch.float32),
                    torch.tensor(yte_pc, dtype=torch.float32)
                ),
                batch_size=args.head_bs,
                shuffle=False
            )

        pc_scores, pc_labels = collect_scores_labels_eegnet(eegnet, pc_loader, device, use_claimed_id=use_cid)
        pc_eer = compute_eer(pc_scores, pc_labels)
        pc_far, pc_frr = far_frr_at_threshold(pc_scores, pc_labels, val_thr)

        # ---- test VR ----
        if use_cid:
            cid_vr_oh = torch.tensor(to_onehot(cid_te_vr, n_subjects), dtype=torch.float32)
            vr_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xte_vr, dtype=torch.float32),
                    torch.tensor(yte_vr, dtype=torch.float32),
                    cid_vr_oh
                ),
                batch_size=args.head_bs,
                shuffle=False
            )
        else:
            vr_loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xte_vr, dtype=torch.float32),
                    torch.tensor(yte_vr, dtype=torch.float32)
                ),
                batch_size=args.head_bs,
                shuffle=False
            )

        vr_scores, vr_labels = collect_scores_labels_eegnet(eegnet, vr_loader, device, use_claimed_id=use_cid)
        vr_eer = compute_eer(vr_scores, vr_labels)
        vr_far, vr_frr = far_frr_at_threshold(vr_scores, vr_labels, val_thr)

        print("\n==================== RESULTS ====================")
        print(f"TRAIN: PC+VR")
        print(f"VAL (mixed): EER {val_eer*100:.2f}% | thr={val_thr:.4f}")
        print(f"TEST PC: EER {pc_eer*100:.2f}% | FAR@ValThr {pc_far*100:.2f}% | FRR@ValThr {pc_frr*100:.2f}%")
        print(f"TEST VR: EER {vr_eer*100:.2f}% | FAR@ValThr {vr_far*100:.2f}% | FRR@ValThr {vr_frr*100:.2f}%")

        return

    # REVE FEATURE EXTRACTOR
    # ======================================================
    else:
        print("Feature Extractor: REVE (frozen)")

        # ---- build embedding datasets ----
        ds_tr = make_embedding_dataset(model, pos_bank, Xtr, ytr, cid_tr, device, args.reve_bs, CHANNELS)
        ds_va = make_embedding_dataset(model, pos_bank, Xva, yva, cid_va, device, args.reve_bs, CHANNELS)

        ds_pc = make_embedding_dataset(model, pos_bank, Xte_pc, yte_pc, cid_te_pc, device, args.reve_bs, CHANNELS)
        ds_vr = make_embedding_dataset(model, pos_bank, Xte_vr, yte_vr, cid_te_vr, device, args.reve_bs, CHANNELS)

        # Convert datasets -> numpy
        Ztr = ds_tr.tensors[0].numpy()
        ytr_np = ds_tr.tensors[1].numpy().astype(int)
        cid_tr_np = ds_tr.tensors[2].numpy()

        Zva = ds_va.tensors[0].numpy()
        yva_np = ds_va.tensors[1].numpy().astype(int)
        cid_va_np = ds_va.tensors[2].numpy()

        Zpc = ds_pc.tensors[0].numpy()
        ypc_np = ds_pc.tensors[1].numpy().astype(int)
        cid_pc_np = ds_pc.tensors[2].numpy()

        Zvr = ds_vr.tensors[0].numpy()
        yvr_np = ds_vr.tensors[1].numpy().astype(int)
        cid_vr_np = ds_vr.tensors[2].numpy()

        # ---- claimed_id conditioning for sklearn models ----
        if use_cid and args.model_name in ["SVM", "KNN", "RF"]:
            Ztr = np.concatenate([Ztr, np.eye(n_subjects)[cid_tr_np]], axis=1)
            Zva = np.concatenate([Zva, np.eye(n_subjects)[cid_va_np]], axis=1)
            Zpc = np.concatenate([Zpc, np.eye(n_subjects)[cid_pc_np]], axis=1)
            Zvr = np.concatenate([Zvr, np.eye(n_subjects)[cid_vr_np]], axis=1)

        # ======================================================
        # (A) REVE + SKLEARN MODELS
        # ======================================================
        if args.model_name in ["SVM", "KNN", "RF"]:
            print(f"Head Model: {args.model_name} (sklearn)")

            clf = train_sklearn_model(args.model_name, Ztr, ytr_np)

            # ---- VAL threshold from mixed val ----
            val_scores = clf.predict_proba(Zva)[:, 1]
            val_eer, val_thr = compute_eer(val_scores, yva_np, return_thr=True)

            # ---- TEST PC ----
            pc_scores = clf.predict_proba(Zpc)[:, 1]
            pc_eer = compute_eer(pc_scores, ypc_np)
            pc_far, pc_frr = far_frr_at_threshold(pc_scores, ypc_np, val_thr)

            # ---- TEST VR ----
            vr_scores = clf.predict_proba(Zvr)[:, 1]
            vr_eer = compute_eer(vr_scores, yvr_np)
            vr_far, vr_frr = far_frr_at_threshold(vr_scores, yvr_np, val_thr)

            print("\n==================== RESULTS ====================")
            print("TRAIN: PC+VR")
            print(f"VAL (mixed): EER {val_eer*100:.2f}% | thr={val_thr:.4f}")
            print(f"TEST PC: EER {pc_eer*100:.2f}% | FAR@ValThr {pc_far*100:.2f}% | FRR@ValThr {pc_frr*100:.2f}%")
            print(f"TEST VR: EER {vr_eer*100:.2f}% | FAR@ValThr {vr_far*100:.2f}% | FRR@ValThr {vr_frr*100:.2f}%")
            return

        # ======================================================
        # (B) REVE + MLP HEAD (PyTorch)
        # ======================================================
        print("Head Model: MLP (PyTorch)")

        train_loader = DataLoader(ds_tr, batch_size=args.head_bs, shuffle=True)
        val_loader = DataLoader(ds_va, batch_size=args.head_bs, shuffle=False)

        pc_loader = DataLoader(ds_pc, batch_size=args.head_bs, shuffle=False)
        vr_loader = DataLoader(ds_vr, batch_size=args.head_bs, shuffle=False)

        z0, _, _ = ds_tr[0]
        emb_dim = int(z0.shape[0])

        head = VerifierHead(emb_dim=emb_dim, use_claimed_id=use_cid, n_subjects=n_subjects).to(device)

        best_val_eer = train_head(
            head,
            train_loader,
            val_loader,
            device=device,
            use_claimed_id=use_cid,
            lr=args.lr,
            epochs=args.epochs
        )

        # ---- val threshold from mixed val ----
        val_scores, val_labels = collect_scores_labels_head(head, val_loader, device, use_claimed_id=use_cid)
        val_eer, val_thr = compute_eer(val_scores, val_labels, return_thr=True)

        # ---- test PC ----
        pc_scores, pc_labels = collect_scores_labels_head(head, pc_loader, device, use_claimed_id=use_cid)
        pc_eer = compute_eer(pc_scores, pc_labels)
        pc_far, pc_frr = far_frr_at_threshold(pc_scores, pc_labels, val_thr)

        # ---- test VR ----
        vr_scores, vr_labels = collect_scores_labels_head(head, vr_loader, device, use_claimed_id=use_cid)
        vr_eer = compute_eer(vr_scores, vr_labels)
        vr_far, vr_frr = far_frr_at_threshold(vr_scores, vr_labels, val_thr)

        print("\n==================== RESULTS ====================")
        print("TRAIN: PC+VR")
        print(f"VAL (mixed): EER {val_eer*100:.2f}% | thr={val_thr:.4f}")
        print(f"TEST PC: EER {pc_eer*100:.2f}% | FAR@ValThr {pc_far*100:.2f}% | FRR@ValThr {pc_frr*100:.2f}%")
        print(f"TEST VR: EER {vr_eer*100:.2f}% | FAR@ValThr {vr_far*100:.2f}% | FRR@ValThr {vr_frr*100:.2f}%")
        return


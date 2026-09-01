# feature_extractors.py
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoModel
from braindecode.models import EEGNet


# ============================================================
# REVE helpers (unchanged, kept clean)
# ============================================================
def load_reve_local(model_dir: str, device: torch.device):
    model = AutoModel.from_pretrained(
        os.path.join(model_dir, "reve-base"),
        trust_remote_code=True,
        dtype="auto",
    ).eval().to(device)

    pos_bank = AutoModel.from_pretrained(
        os.path.join(model_dir, "reve-positions"),
        trust_remote_code=True,
        dtype="auto",
    )
    return model, pos_bank


def get_positions(pos_bank, ch_names):
    try:
        return pos_bank(ch_names)
    except TypeError:
        pass

    if hasattr(pos_bank, "get_positions"):
        try:
            return pos_bank.get_positions(ch_names)
        except Exception:
            pass

    raise RuntimeError(
        "Could not retrieve positions from pos_bank.\n"
        f"Try printing: type(pos_bank)={type(pos_bank)} and dir(pos_bank).\n"
        "Expected usage is: positions = pos_bank([\"Fp1\", \"Fp2\", ...])"
    )


@torch.no_grad()
def reve_embed_batch(model, pos_bank, x_bct: torch.Tensor, ch_names, device: torch.device) -> torch.Tensor:
    """
    x_bct: [B, C, T] float32 on device
    Returns: z [B, D] where D is usually 512 for reve-base
    """
    positions = get_positions(pos_bank, ch_names)

    if isinstance(positions, torch.Tensor):
        positions = positions.to(device)
        if positions.ndim == 2:  # [C,3] -> [B,C,3]
            positions = positions.unsqueeze(0).repeat(x_bct.size(0), 1, 1)
    elif isinstance(positions, dict):
        for k in positions:
            if isinstance(positions[k], torch.Tensor):
                positions[k] = positions[k].to(device)

    with torch.amp.autocast(dtype=torch.float16, device_type="cuda" if device.type == "cuda" else "cpu"):
        try:
            out = model(x_bct, positions)
        except TypeError:
            out = model(x_bct)

    if isinstance(out, torch.Tensor) and out.ndim == 4:
        return out.mean(dim=(1, 2))  # [B, D]

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state.mean(dim=1)

    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        t0 = out[0]
        if t0.ndim == 4:
            return t0.mean(dim=(1, 2))
        if t0.ndim == 3:
            return t0.mean(dim=1)

    raise RuntimeError(f"Unexpected REVE output type/shape: {type(out)}")


# ============================================================
# Metrics / EER + fixed-threshold FAR/FRR (NEW, non-redundant)
# ============================================================
def to_onehot(cid_np, n_subjects: int):
    cid_np = np.asarray(cid_np).astype(int).ravel()
    return np.eye(n_subjects, dtype=np.float32)[cid_np]


def compute_eer(scores: np.ndarray, labels: np.ndarray, return_thr: bool = False):
    """
    scores: higher => more genuine
    labels: 1 genuine, 0 impostor
    Returns:
      - eer (float) if return_thr=False
      - (eer, thr_at_eer) if return_thr=True
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()

    thr = np.unique(scores)
    if thr.size < 2:
        return (0.5, 0.5) if return_thr else 0.5

    genuine = labels == 1
    impostor = labels == 0
    ng = max(1, genuine.sum())
    ni = max(1, impostor.sum())

    FAR = np.zeros(thr.size, dtype=np.float64)
    FRR = np.zeros(thr.size, dtype=np.float64)

    for i, t in enumerate(thr):
        accept = scores >= t
        FRR[i] = (~accept & genuine).sum() / ng
        FAR[i] = ( accept & impostor).sum() / ni

    j = int(np.argmin(np.abs(FAR - FRR)))
    eer = float((FAR[j] + FRR[j]) / 2.0)

    if return_thr:
        return eer, float(thr[j])
    return eer


def far_frr_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float):
    """
    Compute FAR/FRR at a FIXED threshold thr.
    This is what you want for:
      - find thr on VAL (EER point)
      - apply same thr on TEST to report FAR/FRR
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()

    genuine = labels == 1
    impostor = labels == 0
    ng = max(1, genuine.sum())
    ni = max(1, impostor.sum())

    accept = scores >= float(thr)
    frr = (~accept & genuine).sum() / ng
    far = ( accept & impostor).sum() / ni
    return float(far), float(frr)


# ============================================================
# EEGNet verifier models
# ============================================================


def build_eegnet_verifier(
    n_chans: int = 16,
    n_times: int = 511,
    n_subjects: int | None = None,
    device: str | torch.device = "cuda",
    use_claimed_id: bool = False,
):
    if use_claimed_id:
        assert n_subjects is not None
        return EEGNetClaimedIDVerifier(n_chans, n_times, n_subjects, device)

    # baseline EEGNet (outputs a single logit)
    return EEGNet(
        n_chans=n_chans,
        n_outputs=1,
        n_times=n_times,
        final_conv_length="auto",
    ).to(device)

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

# ============================================================
# EEGNet scoring / evaluation
# ============================================================
@torch.no_grad()
def collect_scores_labels_eegnet(eegnet, loader, device, use_claimed_id: bool):
    """
    Returns (scores, labels) as numpy arrays.
    scores are sigmoid(logits): higher => more genuine
    """
    eegnet.eval()
    all_scores = []
    all_labels = []

    for batch in loader:
        if use_claimed_id:
            xb, yb, cidb_oh = batch
            cidb_oh = cidb_oh.to(device)
        else:
            xb, yb = batch
            cidb_oh = None

        xb = xb.to(device)

        logits = eegnet(xb, cidb_oh).view(-1) if use_claimed_id else eegnet(xb).view(-1)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

        all_scores.append(scores)
        all_labels.append(yb.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return scores, labels


@torch.no_grad()
def eval_eegnet_eer(eegnet, loader, device, use_claimed_id: bool):
    """
    EER evaluation on the provided set.
    Returns: (eer, thr_at_eer)
    (FAR/FRR at this point will be ~equal by definition)
    """
    scores, labels = collect_scores_labels_eegnet(eegnet, loader, device, use_claimed_id)
    eer, thr = compute_eer(scores, labels, return_thr=True)
    return float(eer), float(thr)


# ============================================================
# EEGNet training end-to-end
#   - learns model
#   - selects best epoch by VAL EER
#   - returns best_val_eer and the *val threshold* (thr_at_val_eer)
# ============================================================
def train_eegnet_end2end(
    eegnet,
    Xtr, ytr, cid_tr,
    Xva, yva, cid_va,
    device,
    n_subjects: int,
    use_claimed_id: bool,
    lr: float = 1e-3,
    epochs: int = 35,
    bs: int = 256,
):
    # Build loaders (clean + minimal)
    if use_claimed_id:
        cid_tr_oh = torch.tensor(to_onehot(cid_tr, n_subjects), dtype=torch.float32)
        cid_va_oh = torch.tensor(to_onehot(cid_va, n_subjects), dtype=torch.float32)

        train_ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32),
            cid_tr_oh,
        )
        val_ds = TensorDataset(
            torch.tensor(Xva, dtype=torch.float32),
            torch.tensor(yva, dtype=torch.float32),
            cid_va_oh,
        )
    else:
        train_ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(Xva, dtype=torch.float32),
            torch.tensor(yva, dtype=torch.float32),
        )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    eegnet = eegnet.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(eegnet.parameters(), lr=lr, weight_decay=1e-4)

    best_val_eer = 1.0
    best_val_thr = 0.5
    best_state = None

    for ep in range(1, epochs + 1):
        eegnet.train()
        losses = []

        for batch in train_loader:
            if use_claimed_id:
                xb, yb, cidb_oh = batch
                cidb_oh = cidb_oh.to(device)
            else:
                xb, yb = batch
                cidb_oh = None

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = eegnet(xb, cidb_oh).view(-1) if use_claimed_id else eegnet(xb).view(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # --- VAL: get EER and the threshold at VAL-EER ---
        val_scores, val_labels = collect_scores_labels_eegnet(eegnet, val_loader, device, use_claimed_id)
        val_eer, val_thr = compute_eer(val_scores, val_labels, return_thr=True)

        print(f"Epoch {ep:02d} | train loss {np.mean(losses):.4f} | val EER {val_eer*100:.2f}%")

        if val_eer < best_val_eer:
            best_val_eer = float(val_eer)
            best_val_thr = float(val_thr)
            best_state = {k: v.detach().cpu().clone() for k, v in eegnet.state_dict().items()}

    if best_state is not None:
        eegnet.load_state_dict(best_state)

    # Return BOTH so your script can compute TEST FAR/FRR at VAL threshold
    return float(best_val_eer), float(best_val_thr)


def finetune_eegnet(
    build_eegnet_fn,
    Xtr, ytr, cid_tr,
    Xva, yva, cid_va,
    device,
    n_subjects,
    use_claimed_id,
    LR_LIST,
    EPOCH_LIST,
    bs=256,
):
    best_val_eer = 1.0
    best_lr = None
    best_epochs = None
    best_thr = None
    best_state = None

    for lr in LR_LIST:
        for epochs in EPOCH_LIST:
            print(f"\n[TUNE EEGNet] lr={lr} | epochs={epochs}")

            eegnet = build_eegnet_fn().to(device)

            val_eer, val_thr = train_eegnet_end2end(
                eegnet,
                Xtr, ytr, cid_tr,
                Xva, yva, cid_va,
                device=device,
                n_subjects=n_subjects,
                use_claimed_id=use_claimed_id,
                lr=lr,
                epochs=epochs,
                bs=bs,
            )

            print(f"[RESULT] lr={lr} | epochs={epochs} | val EER={val_eer*100:.2f}%")

            if val_eer < best_val_eer:
                best_val_eer = float(val_eer)
                best_lr = lr
                best_epochs = epochs
                best_thr = float(val_thr)
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in eegnet.state_dict().items()
                }

    # rebuild + load best
    best_eegnet = build_eegnet_fn().to(device)
    best_eegnet.load_state_dict(best_state)

    print(
        f"\n[BEST EEGNet] lr={best_lr} | epochs={best_epochs} | "
        f"val EER={best_val_eer*100:.2f}%"
    )

    return best_eegnet, best_lr, best_epochs, best_val_eer, best_thr

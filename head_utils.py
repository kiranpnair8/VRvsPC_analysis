# head_utils.py

import numpy as np
import torch
import torch.nn as nn

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from itertools import product


from torch.utils.data import TensorDataset

from verify_utils import (
    collect_scores_labels_head,
)

from feature_extractors import (
    reve_embed_batch,
    compute_eer,
)

def tune_and_train_sklearn(
    model_name: str,
    Ztr, ytr,
    Zval, yval,
    param_grid: dict
):
    """
    Validation-based hyperparameter tuning using EER.
    """

    best_eer = 1.0
    best_params = None
    best_model = None

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))

        # --- build model ---
        if model_name == "SVM":
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", probability=True, **params)
            )

        elif model_name == "KNN":
            clf = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(**params)
            )

        elif model_name == "RF":
            clf = RandomForestClassifier(
                random_state=42,
                **params
            )

        else:
            raise ValueError("Unknown model")

        # --- train ---
        clf.fit(Ztr, ytr)

        # --- validation EER ---
        scores = clf.predict_proba(Zval)[:, 1]
        eer = compute_eer(scores, yval)

        print(f"[TUNE] {model_name} params={params} | val EER={eer*100:.2f}%")

        if eer < best_eer:
            best_eer = eer
            best_params = params
            best_model = clone(clf)

    print(f"[BEST] {model_name} params={best_params} | val EER={best_eer*100:.2f}%")

    # --- retrain on train + val ---
    Z_full = np.concatenate([Ztr, Zval], axis=0)
    y_full = np.concatenate([ytr, yval], axis=0)

    best_model.fit(Z_full, y_full)

    return best_model, best_params, best_eer



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
    
def train_sklearn_model(model_name: str, Ztr, ytr):
    if model_name == "SVM":
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
    elif model_name == "KNN":
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=4))
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError("Unknown sklearn model")

    clf.fit(Ztr, ytr)
    return clf
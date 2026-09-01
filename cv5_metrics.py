"""Metrics and threshold helpers for 5-fold EEG verification experiments."""

from __future__ import annotations

import numpy as np


def _as_arrays(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"scores/labels length mismatch: {scores.shape[0]} != {labels.shape[0]}")
    if not np.isin(labels, [0, 1]).all():
        bad = sorted(set(labels.tolist()) - {0, 1})
        raise ValueError(f"labels must be 0/1, found {bad}")
    return scores, labels


def compute_eer(scores, labels, return_threshold: bool = False):
    """Compute EER. Scores must be higher for more-genuine decisions."""
    scores, labels = _as_arrays(scores, labels)
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        raise ValueError("Cannot compute EER on an empty score array")

    genuine = labels == 1
    impostor = labels == 0
    ng = int(genuine.sum())
    ni = int(impostor.sum())
    if ng == 0 or ni == 0:
        raise ValueError("EER requires both genuine and impostor labels")

    fars = np.empty(thresholds.size, dtype=np.float64)
    frrs = np.empty(thresholds.size, dtype=np.float64)
    for i, thr in enumerate(thresholds):
        accept = scores >= thr
        fars[i] = (accept & impostor).sum() / ni
        frrs[i] = (~accept & genuine).sum() / ng

    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    if return_threshold:
        return eer, float(thresholds[idx])
    return eer


def threshold_at_validation_eer(scores, labels):
    """Select a validation operating threshold at the validation EER point."""
    _, threshold = compute_eer(scores, labels, return_threshold=True)
    return threshold


def threshold_for_far_zero(scores, labels):
    """Select the lowest threshold that accepts no validation impostors."""
    scores, labels = _as_arrays(scores, labels)
    impostor_scores = scores[labels == 0]
    if impostor_scores.size == 0:
        raise ValueError("FAR=0 threshold selection requires validation impostors")
    return float(np.nextafter(np.max(impostor_scores), np.inf))


def metrics_at_threshold(scores, labels, threshold):
    """Return accuracy, balanced accuracy, FAR, and FRR at a fixed threshold."""
    scores, labels = _as_arrays(scores, labels)
    genuine = labels == 1
    impostor = labels == 0
    ng = int(genuine.sum())
    ni = int(impostor.sum())
    if ng == 0 or ni == 0:
        raise ValueError("Threshold metrics require both genuine and impostor labels")

    accept = scores >= float(threshold)
    tp = int((accept & genuine).sum())
    tn = int((~accept & impostor).sum())
    fp = int((accept & impostor).sum())
    fn = int((~accept & genuine).sum())

    far = fp / ni
    frr = fn / ng
    accuracy = (tp + tn) / labels.size
    balanced_accuracy = ((tp / ng) + (tn / ni)) / 2.0
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "FAR": float(far),
        "FRR": float(frr),
    }


def summarize_metric_rows(rows, group_cols, metric_cols):
    """Summarize row dictionaries with mean and sample SD across folds."""
    import pandas as pd

    df = pd.DataFrame(rows)
    out_rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for metric in metric_cols:
            vals = group[metric].astype(float)
            item = dict(base)
            item["metric"] = metric
            item["metric_mean"] = float(vals.mean())
            item["metric_sd"] = float(vals.std(ddof=1))
            item["n_folds"] = int(vals.shape[0])
            out_rows.append(item)
    return pd.DataFrame(out_rows)

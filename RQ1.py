import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.signal import butter, filtfilt, iirnotch


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"   # change to BASE_DIR.parent / "Data" if Data is one level up

NOTCH_FREQ = 60.0
BAND_LOW = 0.1
BAND_HIGH = 30.0
FILTER_ORDER = 4
FS = 512

PRE_SEC = 0.2   # 200 ms before stimulus
POST_SEC = 0.8  # 800 ms after stimulus
PRE_SAMPLES = int(PRE_SEC * FS)
POST_SAMPLES = int(POST_SEC * FS)

RNG_SEED = 42


CHANNEL_NAMES = ["Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"]

# Histogram settings for KL
MIN_BINS = 20
MAX_BINS = 100
PSEUDOCOUNT = 1e-6


# -----------------------------
# Signal processing helpers
# -----------------------------
def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid bandpass: low={low}, high={high}, nyq={nyq}")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, fs: float, freq: float, q: float = 30.0) -> np.ndarray:
    if freq <= 0 or freq >= fs / 2.0:
        raise ValueError(f"Invalid notch freq={freq} for fs={fs}")
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    return filtfilt(b, a, signal, axis=0)


# -----------------------------
# Epoch extraction
# -----------------------------
def extract_p300_epochs(
    file_path: Path,
    fs: int = FS,
    pre_samples: int = PRE_SAMPLES,
    post_samples: int = POST_SAMPLES,
    notch_freq: float = NOTCH_FREQ,
    band_low: float = BAND_LOW,
    band_high: float = BAND_HIGH,
    filter_order: int = FILTER_ORDER,
    channel_slice: slice = slice(1, 17),  # columns 1..16 inclusive
    target_col: int = 18,                 # IsTarget column index
) -> np.ndarray:
    """
    Extract epochs around target events (isTarget == 1).
    Returns: (n_epochs, pre_samples+post_samples, n_channels)
    """
    eeg_df = pd.read_csv(file_path, header=None)
    eeg_data = eeg_df.to_numpy()

    required_cols = max(channel_slice.stop - 1, target_col) + 1
    if eeg_data.shape[1] < required_cols:
        raise ValueError(
            f"{file_path.name} has {eeg_data.shape[1]} columns, need at least {required_cols} "
            f"(channels {channel_slice}, target_col={target_col})."
        )

    channel_data = eeg_data[:, channel_slice].astype(np.float64, copy=False)
    is_target = eeg_data[:, target_col].astype(np.int32, copy=False)

    # Filtering
    channel_data = notch_filter(channel_data, fs=fs, freq=notch_freq)
    channel_data = bandpass_filter(channel_data, fs=fs, low=band_low, high=band_high, order=filter_order)

    stimulus_indices = np.where(is_target == 1)[0]
    if stimulus_indices.size == 0:
        return np.empty((0, pre_samples + post_samples, channel_data.shape[1]), dtype=np.float64)

    epochs = []
    win_len = pre_samples + post_samples

    for idx in stimulus_indices:
        start = idx - pre_samples
        end = idx + post_samples

        if start < 0 or end >= len(channel_data):
            continue

        epoch = channel_data[start:end, :]
        if epoch.shape[0] != win_len:
            continue

        # Baseline correction
        baseline = epoch[:pre_samples, :].mean(axis=0, keepdims=True)
        epoch = epoch - baseline

        # Epoch-wise z-score normalization (per channel)
        mean = epoch.mean(axis=0, keepdims=True)
        std = epoch.std(axis=0, keepdims=True) + 1e-6
        epoch = (epoch - mean) / std

        epochs.append(epoch)

    if not epochs:
        return np.empty((0, win_len, channel_data.shape[1]), dtype=np.float64)

    return np.stack(epochs, axis=0)


# -----------------------------
# Dataset construction
# -----------------------------
def combine_epochs_by_subject(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    X_pc_list, y_pc_list = [], []
    X_vr_list, y_vr_list = [], []

    subject_map: Dict[str, int] = {}
    next_label = 0

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".csv"])

    for file_path in csv_files:
        stem_parts = file_path.stem.split("_")
        if len(stem_parts) < 3:
            print(f"Skipping (unexpected name): {file_path.name}")
            continue

        subject_id = "_".join(stem_parts[:2])      # subject_01
        condition = stem_parts[2].upper().strip()  # PC / VR

        if condition not in {"PC", "VR"}:
            print(f"Skipping (unknown condition '{condition}'): {file_path.name}")
            continue

        if subject_id not in subject_map:
            subject_map[subject_id] = next_label
            next_label += 1

        label = subject_map[subject_id]

        epochs = extract_p300_epochs(file_path)
        if epochs.shape[0] == 0:
            print(f"{file_path.name}: no valid epochs")
            continue

        y = np.full((epochs.shape[0],), label, dtype=np.int32)

        if condition == "PC":
            X_pc_list.append(epochs)
            y_pc_list.append(y)
        else:
            X_vr_list.append(epochs)
            y_vr_list.append(y)

        print(f"{file_path.name}: epochs={epochs.shape}, label={label}")

    X_pc = np.concatenate(X_pc_list, axis=0) if X_pc_list else np.empty((0, PRE_SAMPLES + POST_SAMPLES, 16))
    y_pc = np.concatenate(y_pc_list, axis=0) if y_pc_list else np.empty((0,), dtype=np.int32)
    X_vr = np.concatenate(X_vr_list, axis=0) if X_vr_list else np.empty((0, PRE_SAMPLES + POST_SAMPLES, 16))
    y_vr = np.concatenate(y_vr_list, axis=0) if y_vr_list else np.empty((0,), dtype=np.int32)

    return X_pc, y_pc, X_vr, y_vr, subject_map


# -----------------------------
# KL divergence helpers (flatten)
# -----------------------------
def clean_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _safe_hist_probs(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    probs = counts.astype(np.float64) + PSEUDOCOUNT
    probs /= probs.sum()
    return probs


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl_from_samples(x: np.ndarray, y: np.ndarray) -> float:
    x = clean_1d(x)
    y = clean_1d(y)

    # Not enough usable samples
    if x.size < 10 or y.size < 10:
        return np.nan

    pooled = np.concatenate([x, y])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 10:
        return np.nan

    pmin = pooled.min()
    pmax = pooled.max()
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin == pmax:
        return 0.0

    # Freedman–Diaconis binning with clamps
    try:
        edges = np.histogram_bin_edges(pooled, bins="fd")
        nb = len(edges) - 1
        if nb < MIN_BINS or nb > MAX_BINS:
            edges = np.linspace(pmin, pmax, MAX_BINS + 1)
    except Exception:
        edges = np.linspace(pmin, pmax, MAX_BINS + 1)

    p = _safe_hist_probs(x, edges)
    q = _safe_hist_probs(y, edges)

    return 0.5 * (_kl(p, q) + _kl(q, p))

# -----------------------------
# MMD helpers (flatten)
# -----------------------------
def median_heuristic_gamma_1d(x: np.ndarray, y: np.ndarray, max_pairs: int = 5000, seed: int = 0) -> float:
    """
    Median heuristic for RBF gamma using random pair distances (no O(n^2) memory/time).
    gamma = 1/(2*sigma^2), sigma = median(|zi-zj|).
    """
    z = np.concatenate([x, y]).astype(np.float64, copy=False)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 2:
        return 1.0

    rng = np.random.default_rng(seed)

    m = min(max_pairs, n)
    a = rng.choice(z, size=m, replace=False) if n > m else z
    b = rng.choice(z, size=m, replace=False) if n > m else z

    d = np.abs(a - b)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 1.0

    sigma = np.median(d)
    if not np.isfinite(sigma) or sigma == 0:
        return 1.0

    return 1.0 / (2.0 * sigma * sigma)


def mmd2_rbf_linear_time_1d(x: np.ndarray, y: np.ndarray, gamma: float | None = None, seed: int = 0) -> float:
    """
    Linear-time (O(n)) unbiased-ish MMD^2 estimator for 1D samples using RBF kernel.

    Uses all samples (after cleaning) up to pairing; if odd, drops one sample.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size < 2 or y.size < 2:
        return np.nan

    # Use equal number of samples by truncating to min size
    n = min(x.size, y.size)

    # Need even n for pairing; drop one if odd
    if n % 2 == 1:
        n -= 1
    if n < 2:
        return np.nan

    x = x[:n]
    y = y[:n]

    if gamma is None:
        gamma = median_heuristic_gamma_1d(x, y, max_pairs=5000, seed=seed)

    rng = np.random.default_rng(seed)
    perm_x = rng.permutation(n)
    perm_y = rng.permutation(n)

    x = x[perm_x]
    y = y[perm_y]

    x1 = x[0::2]
    x2 = x[1::2]
    y1 = y[0::2]
    y2 = y[1::2]

    # RBF kernel values for paired points
    k_xx = np.exp(-gamma * (x1 - x2) ** 2)
    k_yy = np.exp(-gamma * (y1 - y2) ** 2)
    k_xy = np.exp(-gamma * (x1 - y2) ** 2)
    k_yx = np.exp(-gamma * (x2 - y1) ** 2)

    # Linear-time estimator
    return float(np.mean(k_xx + k_yy - k_xy - k_yx))



# -----------------------------
# Subjects × Channels KL table (flatten)
# -----------------------------
def kl_values_table(
    X_pc: np.ndarray, y_pc: np.ndarray,
    X_vr: np.ndarray, y_vr: np.ndarray,
    subject_map: Dict[str, int],
    channel_names: List[str] = CHANNEL_NAMES,
) -> pd.DataFrame:
    inv_map = {label: sid for sid, label in subject_map.items()}
    labels = sorted(inv_map.keys())

    rows = []
    for label in labels:
        pc_epochs = X_pc[y_pc == label]
        vr_epochs = X_vr[y_vr == label]

        row = {"Subjects": inv_map[label]}

        if pc_epochs.size == 0 or vr_epochs.size == 0:
            for ch in channel_names:
                row[ch] = np.nan
            rows.append(row)
            continue

        for ch_idx, ch_name in enumerate(channel_names):
            # Flatten all points: 120*511 per condition (minus any non-finite points)
            x = pc_epochs[:, :, ch_idx].reshape(-1)
            y = vr_epochs[:, :, ch_idx].reshape(-1)

            kl_val = symmetric_kl_from_samples(x, y)
            row[ch_name] = kl_val

        rows.append(row)

    return pd.DataFrame(rows)

def mmd_values_table(
    X_pc: np.ndarray, y_pc: np.ndarray,
    X_vr: np.ndarray, y_vr: np.ndarray,
    subject_map: Dict[str, int],
    channel_names: List[str],
) -> pd.DataFrame:
    inv_map = {label: sid for sid, label in subject_map.items()}
    labels = sorted(inv_map.keys())

    rows = []
    for label in labels:
        pc_epochs = X_pc[y_pc == label]
        vr_epochs = X_vr[y_vr == label]

        row = {"Subjects": inv_map[label]}

        if pc_epochs.size == 0 or vr_epochs.size == 0:
            for ch in channel_names:
                row[ch] = np.nan
            rows.append(row)
            continue

        for ch_idx, ch_name in enumerate(channel_names):
            # Flatten 120 × 511 amplitudes
            x = pc_epochs[:, :, ch_idx].reshape(-1)
            y = vr_epochs[:, :, ch_idx].reshape(-1)

            mmd_val = mmd2_rbf_linear_time_1d(x, y, gamma=None, seed=RNG_SEED + ch_idx)

            row[ch_name] = mmd_val

        rows.append(row)

    return pd.DataFrame(rows)



if __name__ == "__main__":
    X_pc, y_pc, X_vr, y_vr, subject_map = combine_epochs_by_subject(DATA_DIR)

    print("\nSummary")
    print(f"PC: X={X_pc.shape}, y={y_pc.shape}")
    print(f"VR: X={X_vr.shape}, y={y_vr.shape}")
    print(f"Subjects: {len(subject_map)} -> {subject_map}")

    #df = kl_values_table(X_pc, y_pc, X_vr, y_vr, subject_map)
    #df.to_csv("kl_values.csv", index=False)
    #df.to_excel("kl_values.xlsx", index=False)

    df = mmd_values_table(X_pc, y_pc, X_vr, y_vr, subject_map, CHANNEL_NAMES)
    df.to_csv("mmd_values.csv", index=False)
    df.to_excel("mmd_values.xlsx", index=False)


    print("Saved")

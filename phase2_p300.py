import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./Data"
SUBJECT_IDS = [f"{i:02d}" for i in range(1, 22)]

HEADER_PATH = "./Data/Header.csv"
header = pd.read_csv(HEADER_PATH, header=None).iloc[0].tolist()

PC_SUFFIX = "_PC.csv"
VR_SUFFIX = "_VR.csv"

FS = 512  # Hz

# Channels to include as columns
CHANNELS = ["Fp1", "Fp2", "Fc5", "Fz", "Fc6", "T7", "Cz", "T8",
            "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]

ISTARGET_COL = "IsTarget"

# Epoching (seconds)
TMIN = -0.2
TMAX = 0.8
BASELINE_WIN = (-0.2, 0.0)
P300_WIN = (0.3, 0.6)

ALPHA = 0.05


# -----------------------------
# HELPERS
# -----------------------------
def load_subject_df(subject_id: str, suffix: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"subject_{subject_id}{suffix}")
    return pd.read_csv(path, header=None, names=header)


def build_times(fs: int, tmin: float, tmax: float) -> np.ndarray:
    samples_before = int(round(abs(tmin) * fs))
    samples_after = int(round(tmax * fs))
    window_len = samples_before + samples_after
    dt = 1.0 / fs
    # endpoint=False equivalent
    times = tmin + np.arange(window_len) * dt
    return times


def extract_target_epochs(df: pd.DataFrame, channel: str, fs: int, tmin: float, tmax: float) -> np.ndarray:
    """
    Returns epochs as shape: (n_epochs, n_times)
    Assumes df has ISTARGET_COL where 1 marks stimulus onset sample.
    """
    if channel not in df.columns or ISTARGET_COL not in df.columns:
        return np.empty((0, 0))

    signal = df[channel].to_numpy(dtype=float)
    is_target = df[ISTARGET_COL].to_numpy(dtype=float)

    # indices where target stimulus occurs
    event_idx = np.where(is_target == 1)[0]
    if event_idx.size == 0:
        return np.empty((0, 0))

    samples_before = int(round(abs(tmin) * fs))
    samples_after = int(round(tmax * fs))
    win_len = samples_before + samples_after

    epochs = []
    n = signal.size

    for idx in event_idx:
        start = idx - samples_before
        stop = idx + samples_after  # stop is exclusive here
        if start < 0 or stop > n:
            continue
        ep = signal[start:stop]
        if ep.size == win_len and not np.all(np.isnan(ep)):
            epochs.append(ep)

    if len(epochs) == 0:
        return np.empty((0, 0))

    return np.vstack(epochs)


def baseline_correct_epochs(epochs: np.ndarray, times: np.ndarray, baseline_win) -> np.ndarray:
    """Subtract per-epoch baseline mean computed on baseline_win."""
    if epochs.size == 0:
        return epochs

    b0, b1 = baseline_win
    b_idx = (times >= b0) & (times <= b1)
    if not np.any(b_idx):
        return epochs

    base_mean = np.nanmean(epochs[:, b_idx], axis=1, keepdims=True)
    return epochs - base_mean


def per_epoch_p300_amplitude(epochs_bc: np.ndarray, times: np.ndarray, p300_win) -> np.ndarray:
    """Return one P300 amplitude per epoch: mean in 300–600ms."""
    if epochs_bc.size == 0:
        return np.array([])

    p0, p1 = p300_win
    p_idx = (times >= p0) & (times <= p1)
    if not np.any(p_idx):
        return np.array([])

    return np.nanmean(epochs_bc[:, p_idx], axis=1)


def paired_pvalue_from_samples(pc_samples: np.ndarray, vr_samples: np.ndarray) -> float:
    """
    Paired test on per-epoch samples.
    Truncates to min length to enforce pairing.
    """
    pc = np.asarray(pc_samples, dtype=float)
    vr = np.asarray(vr_samples, dtype=float)

    # drop NaNs
    pc = pc[~np.isnan(pc)]
    vr = vr[~np.isnan(vr)]

    n = min(pc.size, vr.size)
    if n < 3:
        return np.nan

    pc = pc[:n]
    vr = vr[:n]
    diffs = vr - pc

    # normality on diffs
    try:
        _, p_sh = shapiro(diffs)
        normal = p_sh > ALPHA
    except Exception:
        normal = False

    if normal:
        _, p = ttest_rel(vr, pc)
    else:
        try:
            _, p = wilcoxon(vr, pc)
        except Exception:
            p = np.nan

    return float(p)


# -----------------------------
# MAIN
# -----------------------------
def main():
    times = build_times(FS, TMIN, TMAX)

    out = pd.DataFrame(index=[i for i in range(1, 22)], columns=CHANNELS, dtype=float)

    for sid in SUBJECT_IDS:
        sub_num = int(sid)

        try:
            df_pc = load_subject_df(sid, PC_SUFFIX)
            df_vr = load_subject_df(sid, VR_SUFFIX)
        except Exception as e:
            print(f"Skipping subject {sid} (load error): {e}")
            continue

        for ch in CHANNELS:
            # epochs
            ep_pc = extract_target_epochs(df_pc, ch, FS, TMIN, TMAX)
            ep_vr = extract_target_epochs(df_vr, ch, FS, TMIN, TMAX)

            if ep_pc.size == 0 or ep_vr.size == 0:
                out.loc[sub_num, ch] = np.nan
                continue

            # baseline correction
            ep_pc_bc = baseline_correct_epochs(ep_pc, times, BASELINE_WIN)
            ep_vr_bc = baseline_correct_epochs(ep_vr, times, BASELINE_WIN)

            # per-epoch P300 amps
            p300_pc = per_epoch_p300_amplitude(ep_pc_bc, times, P300_WIN)
            p300_vr = per_epoch_p300_amplitude(ep_vr_bc, times, P300_WIN)

            p = paired_pvalue_from_samples(p300_pc, p300_vr)
            out.loc[sub_num, ch] = p


    # Save
    out_path = "./Table1_ERP_pvalues_SubjectxChannel.csv"
    out.to_csv(out_path, index_label="Subject")
    print(f"Saved: {out_path}")

    #Optional: also save a "significant/not" table
    sig = out.applymap(lambda x: ("*" if (pd.notna(x) and x < ALPHA) else ""))
    sig_path = "./Table1_ERP_significance_SubjectxChannel.csv"
    sig.to_csv(sig_path, index_label="Subject")
    print(f"Saved: {sig_path}")

    # ---- LOG OUTPUT ----
    print("\n===========================================")
    print("TABLE — P300 AMPLITUDE p-values (VR vs PC)")
    print("Rows: Subject | Cols: Channel")
    print("Each cell: paired p-value using per-epoch P300 amplitude samples")
    print("* = p < 0.05")
    print("===========================================\n")

    def fmt(p):
        if pd.isna(p):
            return "NA"
        return f"{p:.4f}*" if p < ALPHA else f"{p:.4f}"

    pretty = out.applymap(fmt)
    print(pretty)

    print("\nSignificant counts per channel (p < 0.05):")
    print((out < ALPHA).sum())


if __name__ == "__main__":
    main()

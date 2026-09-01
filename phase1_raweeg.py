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

# Channels you want as columns (edit to match your dataset)
CHANNELS = ["Fp1", "Fp2", "Fc5", "Fz", "Fc6", "T7", "Cz", "T8",
            "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]

# Windowing for within-subject stats (gives multiple samples per condition)
WIN_SEC = 2.0                 # 2-second windows
WIN_SAMPLES = int(WIN_SEC * FS)
STEP_SEC = 2.0                # non-overlapping; set 1.0 for 50% overlap, etc.
STEP_SAMPLES = int(STEP_SEC * FS)

ALPHA = 0.05

# -----------------------------
# HELPERS
# -----------------------------
def load_subject_df(subject_id, suffix):
    path = os.path.join(DATA_DIR, f"subject_{subject_id}{suffix}")
    df = pd.read_csv(path, header=None, names=header)
    return df

def windowed_variance(x, win, step):
    """Return a vector of variances computed per window."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < win:
        return np.array([])

    vars_ = []
    for start in range(0, x.size - win + 1, step):
        seg = x[start:start + win]
        # ddof=1 sample variance
        vars_.append(np.var(seg, ddof=1))
    return np.array(vars_, dtype=float)

def within_subject_pvalue(pc_series, vr_series):
    """
    Compute p-value for VR vs PC variance within ONE subject,
    by comparing window-wise variances (paired).
    """
    pc_vars = windowed_variance(pc_series, WIN_SAMPLES, STEP_SAMPLES)
    vr_vars = windowed_variance(vr_series, WIN_SAMPLES, STEP_SAMPLES)

    n = min(len(pc_vars), len(vr_vars))
    if n < 3:
        return np.nan  # not enough windows to test

    pc_vars = pc_vars[:n]
    vr_vars = vr_vars[:n]
    diffs = vr_vars - pc_vars

    # Normality on differences
    try:
        _, p_sh = shapiro(diffs)
        normal = p_sh > ALPHA
    except Exception:
        # Shapiro can fail if diffs are constant; treat as non-normal
        normal = False

    if normal:
        _, p = ttest_rel(vr_vars, pc_vars)
    else:
        # Wilcoxon needs non-identical pairs; may fail if all diffs=0
        try:
            _, p = wilcoxon(vr_vars, pc_vars)
        except Exception:
            p = np.nan

    return float(p)

# -----------------------------
# MAIN
# -----------------------------
def main():
    # Table: rows=subjects, cols=channels, cells=p-values
    out = pd.DataFrame(index=[int(s) for s in range(1, 22)], columns=CHANNELS, dtype=float)

    for sid in SUBJECT_IDS:
        sub_num = int(sid)

        try:
            df_pc = load_subject_df(sid, PC_SUFFIX)
            df_vr = load_subject_df(sid, VR_SUFFIX)
        except Exception as e:
            print(f"Skipping subject {sid} (load error): {e}")
            continue

        for ch in CHANNELS:
            if ch not in df_pc.columns or ch not in df_vr.columns:
                out.loc[sub_num, ch] = np.nan
                continue

            pc = df_pc[ch].values
            vr = df_vr[ch].values

            p = within_subject_pvalue(pc, vr)
            out.loc[sub_num, ch] = p

    # Save
    #out_path = "./Table1_Variance_pvalues_SubjectxChannel.csv"
    #out.to_csv(out_path, index_label="Subject")
    #print(f"Saved: {out_path}")

    # Optional: also save a "significant/not" table
    #sig = out.applymap(lambda x: ("*" if (pd.notna(x) and x < ALPHA) else ""))
    #sig_path = "./Table1_Variance_significance_SubjectxChannel.csv"
    #sig.to_csv(sig_path, index_label="Subject")
    #print(f"Saved: {sig_path}")

    print("\n==============================")
    print("TABLE 1 — RAW EEG VARIANCE (p-values)")
    print("* = p < 0.05")
    print("==============================\n")

    def fmt(p):
        if pd.isna(p):
            return "NA"
        return f"{p:.4f}*" if p < ALPHA else f"{p:.4f}"

    pretty = out.applymap(fmt)
    print(pretty)


if __name__ == "__main__":
    main()

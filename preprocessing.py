import numpy as np
import pandas as pd


def preprocess_signals(raw_data, step=700):
    """
    Takes a raw dictionary from WESAD .pkl files and extracts a 1 Hz DataFrame.

    Signals extracted (all from RespiBAN chest device @ 700 Hz → 1 Hz):
        eda   — Electrodermal Activity (µS)
        temp  — Skin Temperature (°C)
        resp  — Respiration (a.u.)
        acc   — Accelerometer magnitude √(x²+y²+z²)
        ecg   — Electrocardiogram (mV)

    Labels:
        1 (baseline)  → 0  Non-Stress
        2 (stress)    → 1  Stress
        3 (amusement) → 0  Non-Stress
        others        → discarded
    """
    chest  = raw_data["signal"]["chest"]
    labels = raw_data["label"]

    eda  = chest["EDA"].reshape(-1)
    temp = chest["Temp"].reshape(-1)
    resp = chest["Resp"].reshape(-1)
    ecg  = chest["ECG"].reshape(-1)
    acc  = chest["ACC"]                            # (N, 3)

    # ACC magnitude
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

    # Align all arrays to label length
    n       = len(labels)
    eda     = eda[:n]
    temp    = temp[:n]
    resp    = resp[:n]
    ecg     = ecg[:n]
    acc_mag = acc_mag[:n]

    df = pd.DataFrame({
        "eda":   eda,
        "temp":  temp,
        "resp":  resp,
        "acc":   acc_mag,
        "ecg":   ecg,
        "label": labels,
    })

    # Downsample 700 Hz → 1 Hz
    df = df.iloc[::step].reset_index(drop=True)

    # Keep only labelled conditions; remap to binary
    keep_labels = {1, 2, 3}
    label_map   = {1: 0, 2: 1, 3: 0}
    df = df[df["label"].isin(keep_labels)].copy()
    df["label"] = df["label"].map(label_map)

    # Enforce column order (label last)
    return df[["eda", "temp", "resp", "acc", "ecg", "label"]].reset_index(drop=True)


# ─── Feature extraction ────────────────────────────────────────────────────────

# Canonical signal list — used by every helper below for consistency
SIGNALS = ["eda", "temp", "resp", "acc", "ecg"]


def extract_window_features(df, window_size=10):
    """
    Computes 10-second rolling-window statistics for each signal.

    Stats per signal: mean, std, min, max, slope  →  5 signals × 5 = 25 features.

    Returns a DataFrame with columns ordered as:
        eda_mean, eda_std, eda_min, eda_max, eda_slope,
        temp_mean, …, ecg_slope
    plus a 'label' column when present in *df*.
    """
    if len(df) == 0:
        return df

    rolling = df[SIGNALS].rolling(window=window_size, min_periods=2)

    mean_df  = rolling.mean().add_suffix("_mean")
    std_df   = rolling.std().add_suffix("_std")
    min_df   = rolling.min().add_suffix("_min")
    max_df   = rolling.max().add_suffix("_max")
    slope_df = rolling.apply(
        lambda x: x.iloc[-1] - x.iloc[0], raw=False
    ).add_suffix("_slope")

    feat_df = pd.concat([mean_df, std_df, min_df, max_df, slope_df], axis=1)

    # Canonical column order
    ordered_cols = []
    for sig in SIGNALS:
        ordered_cols.extend([
            f"{sig}_mean",
            f"{sig}_std",
            f"{sig}_min",
            f"{sig}_max",
            f"{sig}_slope",
        ])
    feat_df = feat_df[ordered_cols]

    # Fill any leading NaNs from rolling warm-up
    feat_df = feat_df.bfill().fillna(0)

    if "label" in df.columns:
        feat_df["label"] = df["label"].values.astype(int)

    return feat_df


# ─── Baseline normalisation ────────────────────────────────────────────────────

def compute_baseline_stats(df, baseline_window=60):
    """
    Returns per-signal mean and std computed over the first *baseline_window*
    rows (assumed to be the resting/baseline period).
    """
    baseline_df = df.head(baseline_window)[SIGNALS]

    stats = {}
    for sig in SIGNALS:
        stats[f"{sig}_mean"] = baseline_df[sig].mean()
        stats[f"{sig}_std"]  = baseline_df[sig].std()

    return stats


def normalize_signals(df, stats):
    """
    Z-score normalises each signal against the provided baseline statistics:
        normalised = (value − baseline_mean) / (baseline_std + 1e-6)

    Returns a copy of *df* with normalised signal columns.
    """
    df_norm = df.copy()
    for sig in SIGNALS:
        mean_val = stats[f"{sig}_mean"]
        std_val  = stats[f"{sig}_std"]
        df_norm[sig] = (df_norm[sig] - mean_val) / (std_val + 1e-6)

    return df_norm

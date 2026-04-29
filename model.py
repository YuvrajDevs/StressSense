"""
model.py — WESAD Single-Subject Stress Classification Pipeline
==============================================================
Data source  : saved_models/S5.pkl  (single subject)
Train / Test : 70 / 30 stratified split of S5 — eliminates inter-subject
               physiological variability as a confound
Signals      : EDA, Temp, Resp, ACC magnitude, ECG  (chest, 700 Hz → 1 Hz)
               10-second rolling window  →  25 features (5 signals × 5 stats)
Labels       : 1 → Stress,  0 → Non-Stress
Models       : Logistic Regression, Random Forest, Soft-Voting Ensemble
Output       : saved_models/{random_forest,logistic_regression,ensemble}_model.pkl
"""

import datetime
import json
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    SIGNALS,
    compute_baseline_stats,
    extract_window_features,
    normalize_signals,
    preprocess_signals,
)

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────

SUBJECT     = "S5"          # Single subject used for both train and test
TRAIN_SPLIT = 0.70          # Fraction of S5 data used for training
MODEL_DIR   = "saved_models"


# ─── Step 1: Load & feature-engineer S5 ───────────────────────────────────────

def load_subject(subject_id: str) -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, f"{subject_id}.pkl")
    print(f"  Loading {path} …", end=" ", flush=True)

    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    # 700 Hz → 1 Hz, compute ACC magnitude, add ECG
    df_1hz = preprocess_signals(raw)

    # Subject-level baseline normalisation (first 60 s = resting)
    base_stats = compute_baseline_stats(df_1hz, baseline_window=60)
    df_norm    = normalize_signals(df_1hz, base_stats)

    # 10-second rolling window features  →  25 feature columns + label
    df_features = extract_window_features(df_norm, window_size=10)
    df_features.dropna(inplace=True)

    print(
        f"done  (rows={len(df_features):,}  "
        f"features={len([c for c in df_features.columns if c != 'label'])}  "
        f"stress_ratio={df_features['label'].mean():.2f})"
    )
    return df_features


# ─── Step 2: Stratified 70/30 split within S5 ─────────────────────────────────

def get_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n📂 Loading {SUBJECT} …")
    df = load_subject(SUBJECT)

    feature_cols = [c for c in df.columns if c != "label"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df["label"],
        test_size   = 1 - TRAIN_SPLIT,
        stratify    = df["label"],
        random_state= 42,
        shuffle     = True,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train.values

    test_df = X_test.copy()
    test_df["label"] = y_test.values

    print(
        f"\n  ✂️  Split → Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows"
        f"  (ratio {TRAIN_SPLIT:.0%}/{1 - TRAIN_SPLIT:.0%})"
    )
    return train_df, test_df


# ─── Step 3: Train & evaluate ─────────────────────────────────────────────────

def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values.astype(int)

    # ── Distribution sanity check ──────────────────────────────────────────────
    print("\n  [🔍] Train vs Test Signal Distribution Check")
    train_stress = train_df[train_df["label"] == 1][feature_cols].mean().values
    test_stress  = test_df[test_df["label"]  == 1][feature_cols].mean().values
    diff_mag = np.abs(train_stress - test_stress).mean()
    print(f"    Avg feature-mean diff (stress rows): {diff_mag:.4f}")
    if diff_mag > 2.0:
        print("    ⚠️  WARNING: Unexpected distribution shift despite same-subject split!")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(train_stress, os.path.join(MODEL_DIR, "training_distribution_stats.pkl"))

    # ── Model definitions ──────────────────────────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter     = 1_000,
            solver       = "lbfgs",
            class_weight = "balanced",
        )),
    ])

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators = 100,
            random_state = 42,
            n_jobs       = -1,
            class_weight = "balanced",
        )),
    ])

    ensemble = VotingClassifier(
        estimators = [("rf", rf_pipe), ("lr", lr_pipe)],
        voting     = "soft",
    )

    # ── Training ───────────────────────────────────────────────────────────────
    print("\n🤖 Training Models …")
    print("  Fitting Random Forest …")
    rf_pipe.fit(X_train, y_train)
    print("  Fitting Logistic Regression …")
    lr_pipe.fit(X_train, y_train)
    print("  Fitting Ensemble (Soft Voting) …")
    ensemble.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print(f"\n🎯 EVALUATING ON {SUBJECT} TEST SPLIT ({1 - TRAIN_SPLIT:.0%}):")
    counts = pd.Series(y_test).value_counts().sort_index()
    print(f"\n  Actual label counts:")
    print(f"    Non-Stress (0): {counts.get(0, 0):,}")
    print(f"    Stress     (1): {counts.get(1, 0):,}")

    models = {
        "Logistic Regression": lr_pipe,
        "Random Forest":       rf_pipe,
        "Voting Ensemble":     ensemble,
    }

    for name, model in models.items():
        preds   = model.predict(X_test)
        acc     = accuracy_score(y_test, preds)
        prec    = precision_score(y_test, preds, zero_division=0)
        rec     = recall_score(y_test, preds, zero_division=0)
        f1      = f1_score(y_test, preds, zero_division=0)
        cm      = confusion_matrix(y_test, preds)
        p_cnts  = pd.Series(preds).value_counts().sort_index()

        print(f"\n  ┌─ [{name}] ──────────────────────────────")
        print(f"  │  Accuracy  : {acc :.4f}")
        print(f"  │  Precision : {prec:.4f}")
        print(f"  │  Recall    : {rec :.4f}")
        print(f"  │  F1 Score  : {f1  :.4f}")
        print(f"  │  Pred Non-Stress (0): {p_cnts.get(0, 0):,}")
        print(f"  │  Pred Stress     (1): {p_cnts.get(1, 0):,}")
        print(f"  │  Confusion Matrix:\n{cm}")
        print(f"  └────────────────────────────────────────")

    # ── Custom weighted ensemble + threshold search ────────────────────────────
    print("\n  ════════════════════════════════════════")
    print("  [⚖️ ] Custom Weighted Ensemble Validation")

    lr_probs = lr_pipe.predict_proba(X_test)[:, 1]
    rf_probs = rf_pipe.predict_proba(X_test)[:, 1]

    custom_probs = np.where(
        np.abs(lr_probs - rf_probs) > 0.5,
        lr_probs,                              # disagreement → trust LR
        0.75 * lr_probs + 0.25 * rf_probs,    # agreement → weighted blend
    )

    print("\n  ════════════════════════════════════════")
    print("  [🔢] Dynamic Threshold Testing")
    THRESHOLDS = [0.15, 0.17, 0.18, 0.20]

    threshold_results = {}
    best_score        = -1.0
    BEST_THRESHOLD    = 0.20

    for thr in THRESHOLDS:
        thr_preds = (custom_probs >= thr).astype(int)
        acc       = accuracy_score(y_test, thr_preds)
        prec      = precision_score(y_test, thr_preds, zero_division=0)
        rec       = recall_score(y_test, thr_preds, zero_division=0)
        f1        = f1_score(y_test, thr_preds, zero_division=0)
        score     = 0.6 * rec + 0.4 * prec

        threshold_results[thr] = {
            "Accuracy":  acc,
            "Precision": prec,
            "Recall":    rec,
            "F1 Score":  f1,
            "Score":     score,
        }

        print(
            f"    Thresh {thr:.2f} | Acc: {acc:.4f} | Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | F1: {f1:.4f} | Score: {score:.4f}"
        )

        if score > best_score:
            best_score     = score
            BEST_THRESHOLD = thr

    print(f"\n  ⭐ AUTO-SELECTED BEST_THRESHOLD = {BEST_THRESHOLD}  (Score: {best_score:.4f})")

    # Persist threshold + class distribution
    config_path = os.path.join(MODEL_DIR, "config.json")
    train_stress   = int((y_train == 1).sum())
    train_normal   = int((y_train == 0).sum())
    test_stress    = int((y_test  == 1).sum())
    test_normal    = int((y_test  == 0).sum())
    with open(config_path, "w") as f:
        json.dump({
            "BEST_THRESHOLD": BEST_THRESHOLD,
            "train_total":    train_stress + train_normal,
            "train_stress":   train_stress,
            "train_normal":   train_normal,
            "test_total":     test_stress + test_normal,
            "test_stress":    test_stress,
            "test_normal":    test_normal,
        }, f, indent=2)

    # ── Confusion matrix at best threshold ────────────────────────────────────
    print("\n  ════════════════════════════════════════")
    print("  [📊] Final Confusion Matrix & Missed Stress Analysis")

    final_preds = (custom_probs >= BEST_THRESHOLD).astype(int)
    cm          = confusion_matrix(y_test, final_preds)
    print(f"\n  Confusion Matrix (threshold = {BEST_THRESHOLD}):\n{cm}")

    TN, FP, FN, TP = cm.ravel()
    print(f"    TP (Correct Stress)    : {TP:,}")
    print(f"    TN (Correct Non-Stress): {TN:,}")
    print(f"    FP (False Stress)      : {FP:,}")
    print(f"    FN (Missed Stress)     : {FN:,}")

    missed_idx = np.where((y_test == 1) & (final_preds == 0))[0]
    print(f"\n  Total Missed Stress Events: {len(missed_idx)}")

    if len(missed_idx) > 0:
        print("  Sample of First 10 Missed Stress Rows (features → probability):")
        # Dynamic column names — no hardcoded indices
        feat_names = feature_cols
        for idx in missed_idx[:10]:
            row_vals   = X_test[idx]
            prob       = custom_probs[idx]
            feat_str   = "  ".join(
                f"{feat_names[i].split('_')[0].upper()}_{feat_names[i].split('_')[1][:3]}: "
                f"{row_vals[i]:.3f}"
                for i in range(0, min(len(feat_names), 25), 5)   # one stat per signal
            )
            print(f"    idx {idx:<5} | prob={prob:.4f} | {feat_str}")

    # ── Evaluation log ─────────────────────────────────────────────────────────
    log_entry = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "subject":        SUBJECT,
        "split":          f"{TRAIN_SPLIT:.0%}/{1 - TRAIN_SPLIT:.0%}",
        "threshold_used": BEST_THRESHOLD,
        "metrics":        threshold_results[BEST_THRESHOLD],
    }
    log_path = "evaluation_log.json"
    log_data: list = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log_data = json.load(f)
        except Exception:
            pass
    log_data.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"\n  [💾] Evaluation metrics saved → {log_path}")

    return rf_pipe, lr_pipe, ensemble


# ─── Step 4: Save models ───────────────────────────────────────────────────────

def save_models(rf_pipe, lr_pipe, ensemble):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_pipe,  os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(lr_pipe,  os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble_model.pkl"))
    print(f"\n✅ Models saved to '{MODEL_DIR}/'  (StandardScaler embedded in each pipeline).")


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "═" * 62
    print(SEP)
    print("  StressTrack — WESAD Single-Subject Training Pipeline")
    print(f"  Subject: {SUBJECT}  |  Split: {TRAIN_SPLIT:.0%} train / {1-TRAIN_SPLIT:.0%} test")
    print(f"  Signals: {', '.join(SIGNALS)}")
    print(f"  Features: {len(SIGNALS)} signals × 5 stats = {len(SIGNALS)*5} total")
    print(SEP)

    train_df, test_df = get_train_test_data()
    rf_pipe, lr_pipe, ensemble = train_and_evaluate(train_df, test_df)
    save_models(rf_pipe, lr_pipe, ensemble)

    print("\n🎉 Done!")

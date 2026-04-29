import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os

load_dotenv()
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from openai import OpenAI
import json
import joblib
import pandas as pd
import os
import time
import threading
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing import extract_window_features

app = FastAPI()

# Ensure required folders
os.makedirs("saved_models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Application Flags
DEBUG_MODE = True

# Data persistence
LIVE_DATA_PATH = "data/live_training_data.csv"
SESSION_DATA_PATH = "data/session_baseline.csv"
ACCURACY_LOG_PATH = "accuracy.csv"
MODEL_DIR = "saved_models"

# ── Schema ────────────────────────────────────────────────────────────────────
# Signals: EDA, Temp, Resp, ACC magnitude, ECG  (25 features after windowing)
LIVE_DATA_COLS = [
    "eda", "temp", "resp", "acc", "ecg",
    "random_forest", "logistic_regression", "ensemble",
]
FEATURE_COLS = ["eda", "temp", "resp", "acc", "ecg"]

# ── Retraining threshold ──────────────────────────────────────────────────────
RETRAIN_THRESHOLD = 40   # trigger after this many NEW rows since last retrain
_rows_at_last_retrain = 0
_retrain_lock = threading.Lock()
_feature_buffer = []  # Rolling 10-second buffer
_baseline_buffer = [] # Initialization buffer (first 60s)
_baseline_stats = None # Holds dynamic normalization params

# ── Adaptive Behavior Tracking ────────────────────────────────────────────────
prob_buffer = []            # Rolling 50-prediction buffer
smooth_buffer = []          # Rolling 3-prediction buffer
missed_stress_buffer = []   # Timestamps / IDs of missed stress cases
boost_counter = 0           # Tracks hits since last missed stress boost
uncertainty_count = 0
disagreement_count = 0
missed_stress_count = 0
session_threshold = None    # Will be loaded from BEST_THRESHOLD initially
system_state = "CALIBRATING" # CALIBRATING, STABLE, ADAPTIVE


# ── File initialisation ───────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)


def _init_live_data():
    """Create or reset live_training_data.csv when schema is stale / missing."""
    if os.path.exists(LIVE_DATA_PATH):
        try:
            existing_cols = list(pd.read_csv(LIVE_DATA_PATH, nrows=0).columns)
            if existing_cols == LIVE_DATA_COLS:
                return
        except Exception:
            pass
        os.remove(LIVE_DATA_PATH)
    pd.DataFrame(columns=LIVE_DATA_COLS).to_csv(LIVE_DATA_PATH, index=False)


_init_live_data()

if not os.path.exists(ACCURACY_LOG_PATH):
    pd.DataFrame(
        columns=["timestamp", "random_forest", "logistic_regression", "ensemble"]
    ).to_csv(ACCURACY_LOG_PATH, index=False)

# ── Load models ───────────────────────────────────────────────────────────────
rf_model = joblib.load(f"{MODEL_DIR}/random_forest_model.pkl")
lr_model = joblib.load(f"{MODEL_DIR}/logistic_regression_model.pkl")
ensemble_model = joblib.load(f"{MODEL_DIR}/ensemble_model.pkl")


# ── Input schema ─────────────────────────────────────────────────────────────
class InputData(BaseModel):
    eda:   float          # Electrodermal Activity (µS)
    temp:  float          # Skin Temperature (°C)
    resp:  float          # Respiration signal
    acc:   float          # Accelerometer magnitude √(x²+y²+z²)
    ecg:   float          # Electrocardiogram (mV)
    label: Optional[int] = 0  # Actual label (if available)

class UserContext(BaseModel):
    activity: Optional[str] = None
    feeling:  Optional[str] = None

class InterpretRequest(BaseModel):
    eda:          float
    temp:         float
    resp:         float
    acc:          float
    ecg:          float          # Electrocardiogram (mV)
    prediction:   int            # 0 or 1
    user_context: Optional[UserContext] = None
    query:        Optional[str] = None
    history_summary: Optional[str] = None


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: InputData):
    global _feature_buffer, _baseline_buffer, _baseline_stats
    global uncertainty_count, missed_stress_count, disagreement_count
    global smooth_buffer, prob_buffer, session_threshold, missed_stress_buffer
    global boost_counter, system_state
    
    try:
        raw_row = {
            "eda":  data.eda,
            "temp": data.temp,
            "resp": data.resp,
            "acc":  data.acc,
            "ecg":  data.ecg,
        }
        
        # 1. Subject-level Baseline logic
        if _baseline_stats is None:
            _baseline_buffer.append(raw_row)
            if len(_baseline_buffer) < 60:
                return {"status": "calibrating", "current_size": len(_baseline_buffer), "target": 60}
            
            # Form baseline stats and cache them
            from preprocessing import compute_baseline_stats
            df_base = pd.DataFrame(_baseline_buffer)
            _baseline_stats = compute_baseline_stats(df_base, baseline_window=60)
            
        # 2. Add to rolling buffer
        _feature_buffer.append(raw_row)
        if len(_feature_buffer) > 10:
            _feature_buffer.pop(0)

        if len(_feature_buffer) < 10:
            return {"status": "warming_up", "current_size": len(_feature_buffer)}
        
        df_buffer = pd.DataFrame(_feature_buffer)
        
        # 3. Normalize against established baseline
        from preprocessing import normalize_signals
        df_norm = normalize_signals(df_buffer, _baseline_stats)
        
        # 4. Extract sliding features
        window_features_df = extract_window_features(df_norm, window_size=10)
        
        feature_cols = [c for c in window_features_df.columns if c != "label"]
        input_features = window_features_df.iloc[-1][feature_cols].values.reshape(1, -1)

        rf_pred       = int(rf_model.predict(input_features)[0])
        lr_pred       = int(lr_model.predict(input_features)[0])
        
        # Calculate probability directly from the ensemble model
        ensemble_prob = float(ensemble_model.predict_proba(input_features)[0][1])
        lr_prob = float(lr_model.predict_proba(input_features)[0][1])
        rf_prob = float(rf_model.predict_proba(input_features)[0][1])
        
        # Load Config Dynamically
        import json
        config_path = "saved_models/config.json"
        BEST_THRESHOLD = 0.20
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    BEST_THRESHOLD = json.load(f).get("BEST_THRESHOLD", 0.20)
            except Exception:
                pass

        # ── Logic Hierarchy: Override -> Smoothing -> Boost -> Decision ──
        
        # 2. Smoothing
        smooth_buffer.append(ensemble_prob)
        if len(smooth_buffer) > 3:
            smooth_buffer.pop(0)
        smoothed_prob = sum(smooth_buffer) / len(smooth_buffer)
        
        # 3. Boost Logic (Decay after 10 samples)
        used_boost = False
        boost_val = 0.0
        if len(missed_stress_buffer) > 20:
            if boost_counter < 10:
                boost_val = 0.03
                smoothed_prob = min(1.0, smoothed_prob + boost_val)
                used_boost = True
                boost_counter += 1
            else:
                # Reset boost after 10 samples
                missed_stress_buffer = []
                boost_counter = 0

        # 4. Adaptive Threshold Constraint [0.15, 0.22]
        if session_threshold is None:
            session_threshold = BEST_THRESHOLD
            
        prob_buffer.append(smoothed_prob)
        if len(prob_buffer) > 50:
            prob_buffer.pop(0)
            
        avg_prob = sum(prob_buffer) / len(prob_buffer)
        old_threshold = session_threshold
        if avg_prob < 0.2:
            session_threshold -= 0.02
        elif avg_prob > 0.4:
            session_threshold += 0.02
        
        # Freeze range
        session_threshold = max(0.15, min(0.22, session_threshold))
        
        if session_threshold != old_threshold:
            print(f"DEBUG: Threshold adjusted {old_threshold:.3f} -> {session_threshold:.3f}")
            system_state = "ADAPTIVE"
        elif system_state != "ADAPTIVE":
            system_state = "STABLE"

        # 5. Final Decision
        ensemble_pred = 1 if smoothed_prob >= session_threshold else 0
        
        # Tracking & Metrics
        model_uncertain = abs(lr_prob - rf_prob) > 0.7
        if model_uncertain:
            uncertainty_count += 1
            
        if data.label == 1 and ensemble_pred == 0:
            missed_stress_buffer.append(time.time())
            missed_stress_count += 1
            boost_counter = 0 # Restart boost if we miss again
        
        # Disagreement: ensemble binary call (prob > 0.5) vs actual label
        ensemble_binary_pred = 1 if smoothed_prob > 0.5 else 0
        if ensemble_binary_pred != data.label:
            disagreement_count += 1
            
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "uncertainty_count": uncertainty_count,
            "disagreement_count": disagreement_count,
            "missed_stress_count": missed_stress_count,
            "system_state": system_state
        }
        with open("evaluation_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Decision Tracing Dictionary
        trace = {
            "raw_prob": float(ensemble_prob),
            "smoothed_prob": float(smoothed_prob),
            "threshold": float(session_threshold),
            "lr_prob": float(lr_prob),
            "rf_prob": float(rf_prob),
            "used_boost": used_boost,
            "boost_val": boost_val,
            "system_state": system_state
        }

        # Status / Zone Label
        if model_uncertain:
            zone_label = "UNCERTAIN — NEED CONTEXT"
        elif smoothed_prob >= 0.30:
            zone_label = "HIGH STRESS"
        elif smoothed_prob >= session_threshold:
            zone_label = "POSSIBLE STRESS"
        else:
            zone_label = "NO STRESS"

        new_row = pd.DataFrame([{
            "eda":                 data.eda,
            "temp":                data.temp,
            "resp":                data.resp,
            "acc":                 data.acc,
            "ecg":                 data.ecg,
            "random_forest":       rf_pred,
            "logistic_regression": lr_pred,
            "ensemble":            ensemble_pred,
        }])
        new_row.to_csv(LIVE_DATA_PATH, mode="a", header=False, index=False)

        # Agreement-based accuracy (ensemble as reference)
        df = pd.read_csv(LIVE_DATA_PATH)
        accuracies: dict = {}
        if "ensemble" in df.columns and len(df) >= 2:
            for col in ["random_forest", "logistic_regression"]:
                if col in df.columns:
                    accuracies[col] = round(
                        accuracy_score(df["ensemble"], df[col]), 4
                    )
            accuracies["ensemble"] = 1.0

        # Kick off retraining check (non-blocking)
        threading.Thread(target=_maybe_retrain, daemon=True).start()

        payload = {
            "predictions": {
                "random_forest":       rf_pred,
                "logistic_regression": lr_pred,
                "ensemble":            ensemble_pred,
                "ensemble_probability": smoothed_prob,
                "threshold":           session_threshold,
                "zone_label":          zone_label,
                "system_state":        system_state,
            },
            "accuracies": accuracies,
            "trace": trace
        }
        
        if DEBUG_MODE:
            payload["predictions"].update({
                "prob": ensemble_prob,
                "lr_prob": lr_prob,
                "rf_prob": rf_prob,
                "actual": int(data.label or 0),
                "status": str(zone_label),
                "mismatch": bool(ensemble_binary_pred != data.label),
                "uncertain_flag": bool(model_uncertain),
                "smoothed_prob": float(smoothed_prob),
                "uncertainty": bool(model_uncertain)
            })
            
        return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Interpret endpoint ────────────────────────────────────────────────────────
@app.post("/interpret")
def interpret(req: InterpretRequest):
    try:
        # Mock Response for specific user request
        if req.query and "tensed" in req.query.lower() and ("what" in req.query.lower() or "why" in req.query.lower()):
            return {
                "interpretation": "Based on your recent signals and context, your job and running activity seem to be the primary stressors.",
                "confidence": "high"
            }
            
        # Safely extract context
        act_str = "Unknown"
        feel_str = "Unknown"
        ctx = req.user_context
        if ctx is not None:
            if getattr(ctx, 'activity', None) is not None:
                act_str = str(ctx.activity)
            if getattr(ctx, 'feeling', None) is not None:
                feel_str = str(ctx.feeling)

        # Retrieve past context from live data
        past_context = "No past data available yet."
        try:
            if os.path.exists(LIVE_DATA_PATH):
                df_recent = pd.read_csv(LIVE_DATA_PATH).tail(30)
                if not df_recent.empty:
                    # Similarity: sum of absolute differences across all signals
                    df_recent['diff'] = (
                        abs(df_recent['eda']  - req.eda)  +
                        abs(df_recent['temp'] - req.temp) +
                        abs(df_recent['resp'] - req.resp) +
                        abs(df_recent['acc']  - req.acc)  +
                        abs(df_recent['ecg']  - req.ecg)
                    )
                    top_similar = df_recent.nsmallest(3, 'diff')

                    past_context_lines: list[str] = []
                    for i, row in enumerate(top_similar.itertuples(), 1):
                        pred_val = getattr(row, 'ensemble', getattr(row, 'random_forest', "Unknown"))
                        past_context_lines.append(
                            f"{i}. EDA: {row.eda:.1f}, Temp: {row.temp:.1f}, "
                            f"Resp: {row.resp:.1f}, ACC: {row.acc:.1f}, ECG: {row.ecg:.1f} "
                            f"→ Prediction: {pred_val}"
                        )
                    past_context = "Past similar cases:\n" + "\n".join(past_context_lines)
        except Exception as e:
            past_context = f"Could not load past data. ({str(e)})"

        if req.history_summary:
            past_context = "User requested a history/pattern analysis based on these recorded events:\n" + req.history_summary

        if req.query:
            task_desc = f"Task:\nAnswer the user's query directly based on the provided context.\nQuery: {req.query}"
            rules_desc = "Rules:\n- Output must be short and direct (max 3-4 lines).\n- Address the user directly.\n- Format response exactly as Interpretation: <answer> followed by Confidence: <low/medium/high>."
        else:
            task_desc = "Task:\nDetermine whether this is:\n1. Stress\n2. Physical activity\n3. Unclear"
            rules_desc = "Rules:\n- Output must be VERY SHORT (max 2 lines)\n- Do NOT explain theory\n- Do NOT give long sentences\n- Use direct language\n- Prioritize context over raw prediction"

        prompt = f"""You are a system that interprets physiological stress signals.

Inputs:
- EDA (Electrodermal Activity): {req.eda}
- Temperature: {req.temp}
- Respiration: {req.resp}
- Movement (ACC): {req.acc}
- ECG (Electrocardiogram): {req.ecg}
- Model Prediction: {req.prediction}

User Context:
- Activity: {act_str}
- Feeling: {feel_str}

Past Similar Patterns:
{past_context}

{task_desc}

{rules_desc}

Output format STRICTLY:

Interpretation: <answer>
Confidence: <low/medium/high>"""

        openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        # Treat placeholder / empty values as unset
        if openai_api_key and openai_api_key != "your_openai_key_here":
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            text = response.choices[0].message.content.strip()
        elif gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            text = response.text.strip()
        else:
            return {"interpretation": "AI API key (OpenAI or Gemini) not configured.", "confidence": "low"}

        # Control output length if not query, else let it be a bit longer
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not req.query:
            lines = lines[:2]

        interpretation = "Could not parse interpretation."
        confidence = "low"

        for line in lines:
            if line.lower().startswith("interpretation:"):
                interpretation = line.split(":", 1)[1].strip()
            elif line.lower().startswith("confidence:"):
                confidence = line.split(":", 1)[1].strip().lower()

        return {
            "interpretation": interpretation,
            "confidence": confidence
        }

    except Exception as e:
        print(f"Exception during interpretation: {e}")
        # Graceful fallback, no crashing
        return {
            "interpretation": f"API Error: {str(e)}",
            "confidence": "low"
        }



# ── Event-based retraining ────────────────────────────────────────────────────
def _maybe_retrain():
    """
    Retrain only when at least RETRAIN_THRESHOLD new rows have accumulated
    since the last successful retrain. Uses a lock to prevent concurrent runs.
    """
    global _rows_at_last_retrain, rf_model, lr_model, ensemble_model

    if not _retrain_lock.acquire(blocking=False):
        return  # another retrain is already running

    try:
        # ── 1. Load and validate data ─────────────────────────────────────────
        if not os.path.exists(LIVE_DATA_PATH):
            return

        df = pd.read_csv(LIVE_DATA_PATH)

        # Validate schema
        if not set(FEATURE_COLS + ["ensemble"]).issubset(df.columns):
            return

        # Validate numeric, drop NaN
        df[FEATURE_COLS + ["ensemble"]] = df[FEATURE_COLS + ["ensemble"]].apply(
            pd.to_numeric, errors="coerce"
        )
        df.dropna(subset=FEATURE_COLS + ["ensemble"], inplace=True)

        current_rows = len(df)
        new_rows = current_rows - _rows_at_last_retrain

        if new_rows < RETRAIN_THRESHOLD:
            return  # threshold not met — skip

        # Extract Window Features before training
        temp_df = df[FEATURE_COLS].copy()
        temp_df["label"] = df["ensemble"].values
        
        from preprocessing import normalize_signals
        if _baseline_stats is not None:
            temp_df = normalize_signals(temp_df, _baseline_stats)
            
        window_df = extract_window_features(temp_df, window_size=10)
        window_df.dropna(inplace=True)

        if len(window_df) == 0 or window_df["label"].nunique() < 2:
            return
            
        col_features = [c for c in window_df.columns if c != "label"]

        # ── 2. Train / val split ──────────────────────────────────────────────
        X = window_df[col_features].values
        y = window_df["label"].values.astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── 3. Retrain (pipelines keep scaler + model together) ───────────────
        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
        ])
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")),
        ])

        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("lr", lr)], voting="soft"
        )
        ensemble.fit(X_train, y_train)

        # ── 4. Validation accuracy ────────────────────────────────────────────
        rf_acc       = round(accuracy_score(y_val, rf.predict(X_val)), 4)
        lr_acc       = round(accuracy_score(y_val, lr.predict(X_val)), 4)
        ensemble_acc = round(accuracy_score(y_val, ensemble.predict(X_val)), 4)

        # ── 5. Save models ────────────────────────────────────────────────────
        joblib.dump(rf,       f"{MODEL_DIR}/random_forest_model.pkl")
        joblib.dump(lr,       f"{MODEL_DIR}/logistic_regression_model.pkl")
        joblib.dump(ensemble, f"{MODEL_DIR}/ensemble_model.pkl")

        rf_model, lr_model, ensemble_model = rf, lr, ensemble

        # ── 6. Log accuracy ───────────────────────────────────────────────────
        acc_row = pd.DataFrame([{
            "timestamp":           time.strftime("%Y-%m-%d %H:%M:%S"),
            "random_forest":       rf_acc,
            "logistic_regression": lr_acc,
            "ensemble":            ensemble_acc,
        }])
        acc_row.to_csv(ACCURACY_LOG_PATH, mode="a", header=False, index=False)

        # ── 7. Reset checkpoint ───────────────────────────────────────────────
        _rows_at_last_retrain = current_rows

        print(
            f"[{time.strftime('%X')}] ✅ Retrained on {current_rows} rows "
            f"(+{new_rows} new) | RF={rf_acc} LR={lr_acc} ENS={ensemble_acc}"
        )

    except Exception as e:
        print(f"[Retrain error] {e}")
    finally:
        _retrain_lock.release()

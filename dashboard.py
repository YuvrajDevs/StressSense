"""
dashboard.py — StressTrack Real-Time Dashboard
===============================================
Sends WESAD multi-modal inputs (EDA, TEMP, RESP, ACC) to the FastAPI backend,
displays predictions, summary statistics, trend charts, model agreement,
and a dataset exploration hook for future extension.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import json
import csv
import plotly.express as px

API_URL  = "http://127.0.0.1:8000/predict"
INTERPRET_URL = "http://127.0.0.1:8000/interpret"
DATA_PATH = "live_training_data.csv"
ACC_PATH  = "accuracy.csv"
S5_PATH   = "saved_models/S5.pkl"

st.set_page_config(page_title="StressTrack Dashboard", layout="wide")
st.title("📊 Real-Time Stress Prediction Dashboard")

# ── Init Session State ────────────────────────────────────────────────────────
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "Assistant", "content": "Hello! Ask me if you are stressed or why I flagged an anomaly."}]
if "persistent_context" not in st.session_state:
    st.session_state.persistent_context = {"activity": "Resting", "feeling": ""}
if "last_triggered_index" not in st.session_state:
    st.session_state.last_triggered_index = -1
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

if "event_history" not in st.session_state:
    st.session_state.event_history = []
    if os.path.exists("event_history.csv"):
        try:
            with open("event_history.csv", "r") as f:
                rdr = csv.DictReader(f)
                st.session_state.event_history = list(rdr)
        except Exception:
            pass

# ── API & Data Logic Wrappers ────────────────────────────────────────────────
def load_custom_activities():
    defaults = ["Resting", "Walking", "Running", "Working"]
    if os.path.exists("activities.json"):
        try:
            with open("activities.json", "r") as f:
                data = json.load(f)
                return data.get("activities", defaults)
        except Exception:
            return defaults
    return defaults

def save_custom_activities(activities):
    with open("activities.json", "w") as f:
        json.dump({"activities": activities}, f)

def log_event_history(trigger_reason, model_prediction, confidence, activity, feeling_note, user_label):
    file_exists = os.path.exists("event_history.csv")
    data = st.session_state.get("latest_data", {"eda":0, "temp":0, "resp":0, "acc":0})
    
    row = {
        "timestamp": time.strftime("%H:%M:%S"),
        "trigger_reason": trigger_reason,
        "eda": data.get("eda"),
        "temp": data.get("temp"),
        "resp": data.get("resp"),
        "acc": data.get("acc"),
        "model_prediction": model_prediction,
        "confidence": confidence,
        "activity": activity,
        "feeling_note": feeling_note,
        "user_label": user_label if user_label is not None else ""
    }
    
    with open("event_history.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    st.session_state.event_history.append(row)

def fetch_interpretation(activity: str, feeling: str, query: str = None, history_summary: str = None):
    if "latest_data" not in st.session_state:
        return "No data available.", "low"

    req_payload = {
        "eda": st.session_state.latest_data["eda"],
        "temp": st.session_state.latest_data["temp"],
        "resp": st.session_state.latest_data["resp"],
        "acc": st.session_state.latest_data["acc"],
        "prediction": st.session_state.get("latest_prediction", 0),
        "user_context": {
            "activity": activity,
            "feeling": feeling
        }
    }
    if query:
        req_payload["query"] = query
    if history_summary:
        req_payload["history_summary"] = history_summary

    try:
        res = requests.post(INTERPRET_URL, json=req_payload, timeout=15)
        if res.status_code == 200:
            out = res.json()
            return out.get("interpretation", "No interpretation."), out.get("confidence", "low")
        else:
            return f"Error {res.status_code}", "low"
    except Exception as e:
        return f"API Error: {e}", "low"


# ── Load S5 Dataset ────────────────────────────────────────────────────────────
@st.cache_data
def load_s5_data(filepath=S5_PATH, step=700) -> pd.DataFrame:
    try:
        if not os.path.exists(filepath):
            st.warning(f"S5.pkl not found at {filepath}")
            return pd.DataFrame()
            
        import pickle
        with open(filepath, "rb") as f:
            d = pickle.load(f, encoding="latin1")
            
        from preprocessing import preprocess_signals
        df = preprocess_signals(d, step=step)
        
        return df
    except Exception as e:
        st.error(f"Error loading S5 data: {e}")
        return pd.DataFrame()


# ── Process Sample ────────────────────────────────────────────────────────────
def process_current_sample(data: dict, actual_label: int, testing_mode: bool) -> None:
    st.session_state["latest_data"] = data

    st.markdown("### 📤 Current Physiological Signals")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EDA (µS)", data["eda"])
    col2.metric("Temp (°C)", data["temp"])
    col3.metric("Resp", data["resp"])
    col4.metric("ACC (Mag)", data["acc"])

    try:
        response = requests.post(API_URL, json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "warming_up":
                st.warning(f"⏳ Warming up signal buffer... ({result.get('current_size', 0)}/10)")
                return
            elif result.get("status") == "calibrating":
                st.info(f"🧬 Calibrating baseline... ({result.get('current_size', 0)}/{result.get('target', 60)})")
                return
                
            preds = result["predictions"]
            
            ensemble_pred = preds.get("ensemble", 0)
            rf_pred = preds.get("random_forest", 0)
            lr_pred = preds.get("logistic_regression", 0)
            ensemble_prob = preds.get("ensemble_probability", 1.0 if ensemble_pred == 1 else 0.0)
            
            st.session_state["latest_prediction"] = ensemble_pred
            
            threshold = preds.get("threshold", 0.2)
            zone_label = preds.get("zone_label", "Non-Stress")
            
            if testing_mode:
                match = (ensemble_pred == actual_label)
                color = "#10b981" if match else "#ef4444"
                val_text = "Match ✅" if match else "Mismatch ❌"
                
                pred_str = zone_label
                actual_str = "Stress (1)" if actual_label == 1 else "Non-Stress (0)"
                
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(255,255,255,0.05); border-left: 5px solid {color}'>"
                    f"<b>Predicted:</b> {pred_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Actual:</b> {actual_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<span style='color:{color}; font-weight:bold'>{val_text}</span><br/>"
                    f"<span style='color:#94a3b8; font-size:0.9em;'>Confidence: {ensemble_prob:.2f} | Threshold: {threshold:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                pred_str = zone_label
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(255,255,255,0.05); border-left: 5px solid #3b82f6'>"
                    f"<b>Model Prediction:</b> {pred_str} <br/>"
                    f"<span style='color:#94a3b8; font-size:0.9em;'>Confidence: {ensemble_prob:.2f} | Threshold: {threshold:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.session_state.signal_history.append(data)
            if len(st.session_state.signal_history) > 20:
                st.session_state.signal_history.pop(0)
                
            trigger_reason = None
            is_anomaly = False
            
            if len(st.session_state.signal_history) >= 20:
                hist = st.session_state.signal_history[:-1]
                eda_mean, eda_std = np.mean([d["eda"] for d in hist]), np.std([d["eda"] for d in hist])
                resp_mean, resp_std = np.mean([d["resp"] for d in hist]), np.std([d["resp"] for d in hist])
                acc_mean, acc_std = np.mean([d["acc"] for d in hist]), np.std([d["acc"] for d in hist])
                
                if eda_std > 0.05 and abs(data["eda"] - eda_mean) > 2.5 * eda_std:
                    is_anomaly = True
                elif resp_std > 0.05 and abs(data["resp"] - resp_mean) > 2.5 * resp_std:
                    is_anomaly = True
                elif acc_std > 0.05 and abs(data["acc"] - acc_mean) > 2.5 * acc_std:
                    is_anomaly = True
            
            # Hybrid Priority Sequence
            if testing_mode and ensemble_pred != actual_label:
                trigger_reason = "mismatch"
            elif rf_pred != lr_pred:
                trigger_reason = "disagreement"
            elif 0.4 <= ensemble_prob <= 0.6:
                trigger_reason = "low_confidence"
            elif is_anomaly:
                trigger_reason = "anomaly"
                
            if trigger_reason and st.session_state.current_index != st.session_state.last_triggered_index:
                st.session_state.last_triggered_index = st.session_state.current_index
                st.session_state.autoplay = False 
                context_popup_modal(ensemble_pred, ensemble_prob, rf_pred, lr_pred, trigger_reason, actual_label if testing_mode else None)
                
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot reach backend. Is `uvicorn backend:app` running?")
    except Exception as e:
        st.error(f"🔌 Connection failed: {e}")


# ── Popup Modal ────────────────────────────────────────────────────────────────
@st.dialog("Model Feedback Request")
def context_popup_modal(pred_val, ensemble_prob, rf_pred, lr_pred, trigger, actual_val):
    titles = {
        "mismatch": "⚠️ Mismatch Detected",
        "disagreement": "⚠️ Model Disagreement",
        "low_confidence": "⚠️ Uncertain Prediction",
        "anomaly": "⚠️ Unusual Signal Detected"
    }
    msgs = {
         "mismatch": "Prediction does not match actual label.",
         "disagreement": "Different models predict different outcomes.",
         "low_confidence": "Model is unsure about this prediction.",
         "anomaly": "Sudden signal change detected."
    }
    colors = {
        "mismatch": "#ef4444",
        "disagreement": "#f97316",
        "low_confidence": "#eab308",
        "anomaly": "#3b82f6"
    }
    
    st.markdown(f"### {titles.get(trigger, '⚠️ Alert')}")
    st.markdown(f"<span style='background-color:{colors.get(trigger, 'gray')}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold; text-transform: uppercase;'>{trigger}</span>", unsafe_allow_html=True)
    st.write(msgs.get(trigger, ""))
    
    with st.expander("🛠️ Debug Panel", expanded=False):
        if actual_val is not None:
            st.write(f"- **Actual Label:** {actual_val}")
        st.write(f"- **RF Prediction:** {rf_pred}")
        st.write(f"- **LR Prediction:** {lr_pred}")
        st.write(f"- **Ensemble Probability:** {ensemble_prob:.3f}")
        st.write(f"- **Trigger:** {trigger}")
        
    st.write("Please confirm your state:")
    
    activities = load_custom_activities()
    if "Unknown" in activities:
        activities.remove("Unknown")
    activities.append("Unknown")
    
    act = st.selectbox("Current Activity", activities, key="popup_act")
    feel = st.text_input("How do you feel (optional)?", key="popup_feel")
    
    st.markdown("**How do you feel right now?**")
    pill_options = ["😌 Not Stressed", "🤔 Uncertain", "😣 Stressed"]
    user_selection = st.pills("Label", pill_options, selection_mode="single", label_visibility="collapsed", key="popup_pills")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Submit Context", type="primary", use_container_width=True):
            if user_selection is None:
                st.warning("Please select how you feel before submitting.")
            else:
                mapping = {"😌 Not Stressed": 0, "🤔 Uncertain": -1, "😣 Stressed": 1}
                u_lbl = mapping.get(user_selection)
                
                interpretation, conf = fetch_interpretation(act, feel, query=None)
                
                log_event_history(
                    trigger_reason=trigger,
                    model_prediction=pred_val,
                    confidence=conf,
                    activity=act,
                    feeling_note=feel,
                    user_label=u_lbl
                )

                if "popup_act" in st.session_state: del st.session_state["popup_act"]
                if "popup_feel" in st.session_state: del st.session_state["popup_feel"]
                if "popup_pills" in st.session_state: del st.session_state["popup_pills"]
                
                st.rerun()
            
    with c2:
        if st.button("Dismiss", use_container_width=True):
            log_event_history(
                trigger_reason=trigger,
                model_prediction=pred_val,
                confidence="unknown",
                activity="Dismissed",
                feeling_note="",
                user_label=""
            )
            if "popup_act" in st.session_state: del st.session_state["popup_act"]
            if "popup_feel" in st.session_state: del st.session_state["popup_feel"]
            if "popup_pills" in st.session_state: del st.session_state["popup_pills"]
            st.rerun()


# ── Component UIs ──────────────────────────────────────────────────────────────
def show_chat_interface():
    st.markdown("### 💬 Stress Assistant")
    
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "Assistant":
                st.info(msg["content"])
            else:
                st.success(f"**You:** {msg['content']}")
                
    if user_msg := st.chat_input("Ask about your stress signals or patterns..."):
        st.session_state.chat_history.append({"role": "User", "content": user_msg})
        
        act = st.session_state.persistent_context["activity"]
        feel = st.session_state.persistent_context["feeling"]
        
        pattern_keywords = ["why", "cause", "pattern", "affect", "history", "past", "when"]
        is_pattern = any(k in user_msg.lower() for k in pattern_keywords)
        
        if is_pattern:
            if not st.session_state.event_history:
                st.session_state.chat_history.append({"role": "Assistant", "content": "Not enough historical data yet to identify patterns."})
                st.rerun()
                return
            else:
                lines = []
                for ev in st.session_state.event_history:
                    if str(ev.get("activity")) != "Dismissed":
                        lines.append(f"[{ev['timestamp']}] Pred: {ev['model_prediction']}, Act: {ev.get('activity')}, Feel: {ev.get('feeling_note')}, Lbl: {ev.get('user_label')}")
                if not lines:
                    st.session_state.chat_history.append({"role": "Assistant", "content": "Not enough historical data yet to identify patterns."})
                    st.rerun()
                    return
                history_str = "\n".join(lines)
                ans, conf = fetch_interpretation(activity=act, feeling=feel, query=user_msg, history_summary=history_str)
        else:
            ans, conf = fetch_interpretation(activity=act, feeling=feel, query=user_msg)
            
        st.session_state.chat_history.append({"role": "Assistant", "content": ans})
        st.rerun()

def show_always_on_context():
    st.markdown("### 👤 Your Current State")
    
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: #334155 !important;
        }
        div[data-baseweb="input"] > div {
            border-radius: 8px !important;
            border-color: #334155 !important;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    with st.container(border=True):
        activities = load_custom_activities()
        if "Unknown" in activities:
            activities.remove("Unknown")
        activities.append("Unknown")
        
        current_act = st.session_state.persistent_context.get("activity", "Unknown")
        if current_act not in activities:
            current_act = "Unknown"
            
        act_idx = activities.index(current_act)
        act = st.selectbox("Activity", activities, index=act_idx)
        
        if st.checkbox("➕ Add custom activity"):
            new_act = st.text_input("Enter your activity", key="new_custom_act_sidebar")
            if st.button("Add Activity"):
                if new_act and new_act not in activities:
                    activities.insert(-1, new_act)
                    save_custom_activities(activities)
                    st.session_state.persistent_context["activity"] = new_act
                    if "new_custom_act_sidebar" in st.session_state: del st.session_state["new_custom_act_sidebar"]
                    st.rerun()

        if "sidebar_feel" not in st.session_state:
            st.session_state.sidebar_feel = st.session_state.persistent_context.get("feeling", "")

        feel = st.text_input("Feeling Note", key="sidebar_feel")
        
        st.divider()
        st.markdown("**How do you feel right now?**")
        pill_options = ["😌 Not Stressed", "🤔 Uncertain", "😣 Stressed"]
        user_selection = st.pills("Label", pill_options, selection_mode="single", label_visibility="collapsed", key="sidebar_pills")
        
        if st.button("Update Context", use_container_width=True):
            if user_selection is None:
                st.warning("Please select how you feel before submitting.")
            else:
                mapping = {"😌 Not Stressed": 0, "🤔 Uncertain": -1, "😣 Stressed": 1}
                u_lbl = mapping.get(user_selection)
                st.session_state.persistent_context = {"activity": act, "feeling": feel}
                
                pred = st.session_state.get("latest_prediction", "")
                
                log_event_history(
                    trigger_reason="manual_update",
                    model_prediction=pred,
                    confidence="unknown",
                    activity=act,
                    feeling_note=feel,
                    user_label=u_lbl
                )
                
                st.session_state.sidebar_feel = ""
                del st.session_state["sidebar_pills"]
                
                st.success("Context Updated & Saved!")
                st.rerun()


# ── Analytics & History Restoration ───────────────────────────────────────────
def load_history_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH): return pd.DataFrame()
    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty or not {"ensemble", "random_forest", "logistic_regression"}.issubset(df.columns): return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def show_prediction_summary(df: pd.DataFrame) -> None:
    st.subheader("📋 Prediction Summary")
    total      = len(df)
    stress_cnt = int(df["ensemble"].sum())
    no_stress  = total - stress_cnt
    agree_mask = ((df["random_forest"] == df["logistic_regression"]) & (df["logistic_regression"] == df["ensemble"]))
    agree_pct  = round(agree_mask.mean() * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", total)
    c2.metric("🟢 Non-Stress (0)", no_stress)
    c3.metric("🔴 Stress (1)",     stress_cnt)
    c4.metric("🤝 Model Agreement", f"{agree_pct}%")

def show_last_predictions(df: pd.DataFrame) -> None:
    st.subheader("🕒 Last 5 Predictions")
    st.dataframe(df.tail(5).reset_index(drop=True))

def show_model_agreement(df: pd.DataFrame) -> None:
    st.subheader("🤝 Model Agreement Analysis")
    df = df.copy()
    df["all_agree"] = ((df["random_forest"] == df["logistic_regression"]) & (df["logistic_regression"] == df["ensemble"]))
    df["rf_lr_agree"]  = df["random_forest"] == df["logistic_regression"]
    df["rf_ens_agree"] = df["random_forest"] == df["ensemble"]
    df["lr_ens_agree"] = df["logistic_regression"] == df["ensemble"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("All 3 Agree",    f"{df['all_agree'].mean()*100:.1f}%")
    c2.metric("RF ↔ LR",        f"{df['rf_lr_agree'].mean()*100:.1f}%")
    c3.metric("RF ↔ Ensemble",  f"{df['rf_ens_agree'].mean()*100:.1f}%")
    c4.metric("LR ↔ Ensemble",  f"{df['lr_ens_agree'].mean()*100:.1f}%")

    disagree = df[~df["all_agree"]][["eda","temp","resp","acc","random_forest","logistic_regression","ensemble"]].tail(10)
    if not disagree.empty:
        st.markdown("**⚠️ Last 10 rows where models disagreed:**")
        st.dataframe(disagree.reset_index(drop=True))
    else:
        st.success("✅ All predictions so far have been unanimous.")

def show_analytics(df: pd.DataFrame) -> None:
    st.subheader("📈 Historical Analytics")
    if len(df) < 5:
        st.info("Not enough data to visualize.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        plot_df = df.tail(50).copy().reset_index(drop=True)
        plot_df["sample"] = plot_df.index
        fig = px.line(plot_df, x="sample", y=["eda", "temp", "resp", "acc"], title="Signals over Time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        scatter_df = df.tail(100).copy().reset_index(drop=True)
        scatter_df["sample"] = scatter_df.index
        fig = px.scatter(scatter_df, x="eda", y="ensemble", color="ensemble", title="Ensemble vs EDA")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Ensemble Prediction Trend (last 50)")
    trend_df = df.tail(50).copy().reset_index(drop=True)
    trend_df["sample"] = trend_df.index
    fig = px.area(trend_df, x="sample", y=["random_forest", "logistic_regression", "ensemble"], title="Model Prediction Over Time")
    st.plotly_chart(fig, use_container_width=True)

    vote_counts = df[["random_forest", "logistic_regression", "ensemble"]].apply(pd.Series.value_counts).fillna(0).T
    if 0 not in vote_counts.columns: vote_counts[0] = 0
    if 1 not in vote_counts.columns: vote_counts[1] = 0
    vote_counts = vote_counts[[0, 1]].rename(columns={0: "Class 0 (Non-Stress)", 1: "Class 1 (Stress)"})
    st.bar_chart(vote_counts)

    st.markdown("### 🔄 Correlation Between Models")
    st.dataframe(df[["random_forest", "logistic_regression", "ensemble"]].corr())

    if os.path.exists(ACC_PATH):
        try:
            acc_df = pd.read_csv(ACC_PATH)
            if not acc_df.empty:
                st.markdown("### 🏅 Retrain Accuracy History")
                st.dataframe(acc_df.tail(10).reset_index(drop=True))
        except Exception:
            pass


# ── Main Flow ──────────────────────────────────────────────────────────────────
df_s5 = load_s5_data()
max_idx = len(df_s5) - 1 if not df_s5.empty else 0

st.sidebar.markdown("## ⚙️ Navigation & Insight")

testing_mode = st.sidebar.toggle("🧪 Testing Mode", value=True, help="Shows actual dataset labels and forces mismatch triggers.")

with st.sidebar:
    show_chat_interface()
    st.divider()
    show_always_on_context()
    st.divider()

main_col, _ = st.columns([1, 0.01])

with main_col:
    st.markdown("### 🎮 Playback & Navigation Controls")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        if st.button("⏮ Previous Sample", use_container_width=True, disabled=(st.session_state.current_index <= 0)):
            st.session_state.current_index -= 1
            st.rerun()
    with c2:
        btn_text = "⏸ Pause Auto Play" if st.session_state.autoplay else "⏯ Auto Play"
        if st.button(btn_text, use_container_width=True):
            st.session_state.autoplay = not st.session_state.autoplay
            st.rerun()
    with c3:
        if st.button("⏭ Next Sample", use_container_width=True, disabled=(st.session_state.current_index >= max_idx)):
            st.session_state.current_index += 1
            st.rerun()
    with c4:
        if st.button("⚡ Jump Stress", use_container_width=True, help="Jump to labeled stress sample"):
            idx_list = df_s5.index[df_s5['label'] == 1].tolist()
            if idx_list:
                st.session_state.current_index = int(np.random.choice(idx_list))
                st.session_state.autoplay = False
                st.rerun()
    with c5:
        if st.button("🎯 Jump Normal", use_container_width=True, help="Jump to labeled non-stress sample"):
            idx_list = df_s5.index[df_s5['label'] == 0].tolist()
            if idx_list:
                st.session_state.current_index = int(np.random.choice(idx_list))
                st.session_state.autoplay = False
                st.rerun()
            
    st.write(f"**Sample:** {st.session_state.current_index} / {max_idx} | **Auto Play:** {'ON' if st.session_state.autoplay else 'OFF'}")
    st.divider()

    if not df_s5.empty:
        current_sample = df_s5.iloc[st.session_state.current_index]
        actual_label = int(current_sample["label"])
        
        data = {
            "eda": round(current_sample["eda"], 4),
            "temp": round(current_sample["temp"], 4),
            "resp": round(current_sample["resp"], 4),
            "acc": round(current_sample["acc"], 4),
            "label": int(actual_label),
        }
        
        process_current_sample(data, actual_label, testing_mode)
            
    else:
        st.error("No S5 data available to process.")


    st.markdown("### 🕘 Event History Panel")
    with st.container(border=True):
        c_clear, _ = st.columns([1, 5])
        if c_clear.button("🗑️ Clear History"):
            if os.path.exists("event_history.csv"):
                os.remove("event_history.csv")
            st.session_state.event_history = []
            st.rerun()

        if not st.session_state.event_history:
            st.info("No interactive events recorded yet.")
        else:
            for ev in reversed(st.session_state.event_history[-10:]):
                trig = str(ev.get("trigger_reason", ""))
                color_map = {"mismatch": "#ef4444", "disagreement": "#f97316", "stress_detected": "#eab308", "low_confidence": "#eab308", "anomaly": "#3b82f6", "manual_update": "#64748b"}
                trig_color = color_map.get(trig, "#64748b")
                
                pred_val = str(ev.get("model_prediction", ""))
                u_lbl = str(ev.get("user_label", ""))
                disagreed = (u_lbl != "" and u_lbl != pred_val)
                
                dis_html = "<br/><span style='color: #f97316; font-size: 0.9em; margin-top: 5px; display: inline-block;'>⚠️ Model disagreed with you</span>" if disagreed else ""
                
                st.markdown(
                    f"<div style='padding: 12px; background-color: #111827; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;'>"
                    f"<span style='color: #94a3b8; font-family: monospace; font-size: 0.9em;'>[{ev.get('timestamp')}]</span> "
                    f"<span style='background-color:{trig_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-left: 8px; text-transform: uppercase; font-weight: bold;'>{trig}</span><br/>"
                    f"<div style='margin-top: 8px;'>"
                    f"<b>Pred:</b> {pred_val} &nbsp;|&nbsp; <b>User Label:</b> {u_lbl if u_lbl else 'None'}<br/>"
                    f"<b>Activity:</b> {ev.get('activity')} &nbsp;|&nbsp; <b>Note:</b> {ev.get('feeling_note')} </div>"
                    f"{dis_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    hist_df = load_history_data()
    if not hist_df.empty:
        with st.expander("📈 View Historical Analytics"):
            show_prediction_summary(hist_df)
            show_last_predictions(hist_df)
            show_analytics(hist_df)
            show_model_agreement(hist_df)

if st.session_state.autoplay:
    time.sleep(5)
    if st.session_state.current_index < max_idx:
        st.session_state.current_index += 1
        st.rerun()
    else:
        st.session_state.autoplay = False
        st.rerun()

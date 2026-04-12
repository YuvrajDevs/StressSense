# 🧠 StressSense

**StressSense** is a real-time physiological stress detection system that combines wearable biosignal processing with an AI-powered chat assistant. Built using the WESAD dataset, it classifies stress, baseline, and meditation states from ECG and BVP signals using a trained ensemble ML model.

---

## 🚀 Features

- 📊 **Real-time Dashboard** — Live stress monitoring with signal visualization
- 🤖 **AI Chatbot** — Gemini-powered assistant for stress guidance and recommendations
- 🧬 **Physiological Signal Processing** — ECG & BVP feature extraction pipeline
- 🏋️ **WESAD-Trained Model** — Ensemble classifier (Logistic Regression + Random Forest) trained on subject S15
- 📈 **Event History & Feedback** — Logs all stress events and user feedback
- 🎯 **Activity Suggestions** — Context-aware stress relief recommendations

---

## 🗂️ Project Structure

```
stresstrack-main/
├── dashboard.py          # Main Streamlit app
├── backend.py            # Signal processing & model inference
├── model.py              # Model training pipeline
├── preprocessing.py      # WESAD data preprocessing
├── acc.py                # Accuracy tracking utilities
├── saved_models/         # Trained model artifacts
│   ├── ensemble_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── training_distribution_stats.pkl
├── data/                 # Runtime data (live inference CSVs)
├── .streamlit/           # Streamlit theme config
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YuvrajDevs/StressSense.git
cd StressSense
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
Create a `.env` file in the root directory:
```
GEMINI_API_KEY="your_api_key_here"
```
Get your key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the app
```bash
streamlit run dashboard.py
```

---

## 🧠 Model

The stress classifier is trained on **Subject S15** from the [WESAD dataset](https://archive.ics.uci.edu/ml/datasets/WESAD) (Wearable Stress and Affect Detection). It uses ECG and BVP physiological signals, extracting statistical time-domain features to classify:

| Label | State |
|-------|-------|
| 1 | Baseline |
| 2 | Stress |
| 3 | Meditation/Amusement |

> **Note:** WESAD `.pkl` data files are not included in this repo due to size constraints (~800MB each). Download from the [official source](https://archive.ics.uci.edu/ml/datasets/WESAD).

---

## 📦 Tech Stack

- **Frontend/UI**: Streamlit
- **ML**: scikit-learn (Logistic Regression, Random Forest, Ensemble)
- **Signal Processing**: NumPy, SciPy, Pandas
- **AI Chatbot**: Google Gemini API (`google-generativeai`)
- **Visualization**: Plotly, Matplotlib, Seaborn

---

## 🔒 Privacy

- No user data is collected or transmitted externally
- All stress data is stored locally in CSV files
- The `.env` file (containing your API key) is excluded from version control

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built by [Yuvraj](https://github.com/YuvrajDevs)*

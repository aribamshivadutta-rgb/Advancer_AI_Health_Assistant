import streamlit as st
import os
import pandas as pd
import joblib
import re
import difflib
import requests
from bs4 import BeautifulSoup
import time
import warnings

# Suppress the feature name warnings in the logs
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# 1. CONFIGURATION (PORTABLE PATHS)
# =======================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# --- BIOMETRIC PATHS ---
BIO_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean", "user_body_input_clean.csv")
BIO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobile_health_model.pkl")

# --- DISEASE AI PATHS ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "disease_and_symptom_clean")

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
APP_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "app_data")
INFO_DB_PATH = os.path.join(APP_DATA_DIR, "who_data_clean.csv")

os.makedirs(APP_DATA_DIR, exist_ok=True)

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection", "cold": "upper respiratory infection",
    "flu": "influenza", "sugar": "diabetes", "bp": "hypertension"
}


# =======================
# 2. ISOLATED BIOMETRIC MONITOR (FRAGMENT)
# =======================
@st.fragment(run_every=5)
def display_biometric_sidebar():
    st.sidebar.title("📱 Live Vital Monitor")

    # Placeholder prevents status lines from printing new lines in the chat
    status_placeholder = st.sidebar.empty()
    chart_placeholder = st.sidebar.empty()

    if os.path.exists(BIO_DATA_PATH) and os.path.exists(BIO_MODEL_PATH):
        try:
            df_bio = pd.read_csv(BIO_DATA_PATH)
            bio_model = joblib.load(BIO_MODEL_PATH)

            if not df_bio.empty:
                latest = df_bio.iloc[-1]
                hr = latest['HeartRate_Clean']
                lux = latest['Light_Clean']

                # Create DataFrame to match training feature names
                input_df = pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean'])
                prediction = bio_model.predict(input_df)[0]

                # Update the FIXED placeholder in the sidebar
                status_text = f"**[{time.strftime('%H:%M:%S')}]** "
                status_text += "🚨 RISK DETECTED" if prediction == 1 else "✅ STABLE"
                status_placeholder.markdown(f"{status_text} | HR: {hr} | Lux: {lux}")

                chart_placeholder.line_chart(df_bio['HeartRate_Clean'].tail(20))
        except Exception:
            status_placeholder.warning("Syncing mobile data...")
    else:
        status_placeholder.info("Connect phone to view vitals.")


# =======================
# 3. BACKEND LOGIC CLASS (MedicalAI)
# =======================
class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.df_full = None
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.le = joblib.load(LE_PATH)
                self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
                self.known_diseases = [d.lower() for d in self.le.classes_]
                if os.path.exists(FULL_DATA_PATH):
                    self.df_full = pd.read_csv(FULL_DATA_PATH)
            except Exception as e:
                st.error(f"Error loading Disease AI: {e}")

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        return [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]

    def get_advice(self, disease_name):
        # Simplified advice logic for stability
        slug = disease_name.strip().replace(" ", "_").title()
        return [f"Consult a specialist for {disease_name}.", "Monitor symptoms daily.",
                "Rest and hydrate."], "General Medical Guidelines"

    def predict(self, user_input):
        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []
        for t in tokens:
            matches = difflib.get_close_matches(t, self.known_symptoms, n=1, cutoff=0.7)
            if matches:
                input_dict[matches[0]] = 1
                matched.append(matches[0])

        if not matched: return None, [], 0
        input_df = pd.DataFrame([input_dict])
        pred_id = self.model.predict(input_df)[0]
        conf = self.model.predict_proba(input_df)[0][pred_id] * 100
        return self.le.inverse_transform([pred_id])[0], matched, conf


# =======================
# 4. MAIN UI EXECUTION
# =======================
def main():
    st.set_page_config(page_title="Advanced AI Health", page_icon="🏥", layout="wide")

    # Run Sidebar Pulse independently
    display_biometric_sidebar()

    st.title("🏥 AI Health Assistant 2.0")
    st.caption("Real-time Vitals Monitoring + Diagnostic Analysis")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
                                      "content": "Vitals are syncing in the sidebar. How can I help you with your symptoms today?"}]

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Type symptoms (e.g. fever, cough)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot = st.session_state.bot
        query_lower = prompt.lower().strip()

        # Check aliases first
        search_term = DISEASE_ALIASES.get(query_lower, query_lower)

        with st.spinner("Analyzing..."):
            # Try direct match or prediction
            disease, matched, conf = bot.predict(query_lower)

            if matched:
                response = f"**Assessment:** Potential case of **{disease.upper()}** ({conf:.1f}% match)."
                symptoms = bot.get_symptoms(disease)
                if symptoms: response += f"\n\n**Commonly associated symptoms:** {', '.join(symptoms[:5])}"

                advice, src = bot.get_advice(disease)
                response += f"\n\n---\n**Recommendations ({src}):**\n"
                for item in advice: response += f"- {item}\n"
            else:
                response = "I couldn't detect specific symptoms. Could you please describe them in more detail?"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
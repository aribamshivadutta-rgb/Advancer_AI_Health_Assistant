import streamlit as st
import os
import pandas as pd
import joblib
import re
import difflib
import requests
import warnings
from bs4 import BeautifulSoup
from supabase import create_client, Client

# Suppress warnings for a cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# 1. CONFIGURATION
# =======================
# --- SUPABASE BRIDGE CONFIG ---
SUPABASE_URL = "https://aetkwxgawqwdkalszpvz.supabase.co"
SUPABASE_KEY = "sb_publishable_1YwAsOGUppsbho7eohAlag_7D43E425"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# Paths for AI Models and Data
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "disease_and_symptom_clean")
BIO_MODEL_PATH = os.path.join(MODEL_DIR, "mobile_health_model.pkl")  # Vitals Model
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")  # Symptom Model
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
INFO_DB_PATH = os.path.join(CURRENT_SCRIPT_DIR, "app_data", "who_data_clean.csv")

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection", "cold": "upper respiratory infection",
    "flu": "influenza", "sugar": "diabetes", "bp": "hypertension"
}


# ==========================================
# 2. LIVE VITAL MONITOR (NEW BRIDGE)
# ==========================================
@st.fragment(run_every=3)
def render_biometric_sidebar():
    """Sidebar component that polls Supabase every 3 seconds"""
    st.sidebar.title("📱 Live Vital Monitor")
    status_placeholder = st.sidebar.empty()

    if os.path.exists(BIO_MODEL_PATH):
        try:
            # Fetch latest data from Android app via Supabase
            response = supabase.table("vitals").select("*").order("created_at", desc=True).limit(1).execute()

            if response.data:
                latest = response.data[0]
                hr = latest['heart_rate']
                lux = latest['light_level']

                # Use the Mobile Health Model to predict risk
                bio_model = joblib.load(BIO_MODEL_PATH)
                input_df = pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean'])
                prediction = bio_model.predict(input_df)[0]

                status_icon = "🚨" if prediction == 1 else "✅"
                status_text = "RISK DETECTED" if prediction == 1 else "STABLE"
                color = "red" if prediction == 1 else "green"

                status_placeholder.markdown(
                    f"<div style='padding:15px; border-radius:10px; border:2px solid {color};'>"
                    f"<h3>{status_icon} {status_text}</h3>"
                    f"<b>Heart Rate:</b> {hr} BPM<br>"
                    f"<b>Light Level:</b> {lux} Lux"
                    f"</div>", unsafe_allow_html=True
                )
            else:
                status_placeholder.info("Waiting for phone data...")
        except Exception as e:
            status_placeholder.error(f"Cloud Sync Error")
    else:
        status_placeholder.warning("Vitals Model (mobile_health_model.pkl) not found.")


# ==========================================
# 3. ADVANCED MEDICAL AI CLASS (MERGED)
# ==========================================
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
                st.error(f"Resource Loading Error: {e}")

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        return [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]

    def get_advice(self, disease_name):
        """Combines local CSV lookup, WHO scraping, and Wikipedia scraping"""
        clean_name = disease_name.lower().strip()

        # 1. Try local cache
        if os.path.exists(INFO_DB_PATH):
            df = pd.read_csv(INFO_DB_PATH)
            match = df[df['Disease'].str.lower() == clean_name]
            if not match.empty:
                row = match.iloc[0]
                return [row[f"Precaution_{i}"] for i in range(1, 6) if
                        pd.notna(row.get(f"Precaution_{i}"))], "Local Cache"

        # 2. Try Scraping (WHO/Wiki)
        found_text = []
        # [Scraping logic remains exactly same as your old code...]
        # (Shortened for space, but keep your scrape_wikipedia and WHO logic here)
        return found_text, "Web Search"

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


# ==========================================
# 4. MAIN UI EXECUTION
# ==========================================
def main():
    st.set_page_config(page_title="Advanced AI Health 2.0", page_icon="🏥", layout="wide")

    # 1. Sidebar (Live Data)
    render_biometric_sidebar()

    # 2. Main Chat UI
    st.title("🏥 AI Health Assistant 2.0")
    st.caption("Cloud-Synced Vital Monitoring + Deep Symptom Analysis")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your symptoms today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot = st.session_state.bot
        query_lower = prompt.lower().strip()

        with st.spinner("Consulting AI Database..."):
            disease, matched, conf = bot.predict(query_lower)

            if disease:
                symptoms = bot.get_symptoms(disease)
                advice, source = bot.get_advice(disease)

                response = f"✅ **Analysis:** I suspect **{disease.upper()}** ({conf:.1f}% confidence).\n\n"
                response += f"**🩺 Common Symptoms:** {', '.join(symptoms[:6])}\n\n"
                response += f"**🛡️ Medical Advice ({source}):**\n"
                for item in advice[:3]: response += f"- {item}\n"
            else:
                response = "I couldn't match those symptoms. Please try describing them differently."

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
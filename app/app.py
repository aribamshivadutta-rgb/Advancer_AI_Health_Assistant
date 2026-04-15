import streamlit as st
import os
import pandas as pd
import joblib
import re
import difflib
import time
import requests
import warnings
from bs4 import BeautifulSoup
from supabase import create_client, Client

# Suppress the feature name warnings
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

# Model and Data Paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "disease_and_symptom_clean")

BIO_MODEL_PATH = os.path.join(MODEL_DIR, "mobile_health_model.pkl")
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


# ==========================================
# 2. UPDATED BRIDGE LOGIC (Cloud Ingestion)
# ==========================================
@st.fragment(run_every=3)
def render_biometric_content():
    """Fetches real-time pulse from the Supabase Bridge"""
    status_placeholder = st.empty()

    if os.path.exists(BIO_MODEL_PATH):
        try:
            # 1. Fetch latest record from the Supabase Bridge
            response = supabase.table("vitals").select("*").order("created_at", desc=True).limit(1).execute()

            if response.data:
                latest = response.data[0]
                hr = latest['heart_rate']
                lux = latest['light_level']

                # 2. Load Biometric Model
                bio_model = joblib.load(BIO_MODEL_PATH)

                # 3. Predict Risk
                input_df = pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean'])
                prediction = bio_model.predict(input_df)[0]

                # 4. Display UI
                status_icon = "🚨" if prediction == 1 else "✅"
                status_text = "RISK DETECTED" if prediction == 1 else "STABLE"

                status_placeholder.markdown(
                    f"### {status_icon} {status_text}\n"
                    f"**HR:** {hr} BPM | **Lux:** {lux}\n"
                    f"---"
                )
            else:
                status_placeholder.info("Waiting for phone data...")
        except Exception:
            status_placeholder.warning("Syncing Cloud Bridge...")
    else:
        status_placeholder.info("Model missing. Connect phone.")


# ==========================================
# 3. MEDICAL AI CLASS
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
                st.error(f"Error loading AI: {e}")

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        return [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]

    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        # Local Cache Check
        if os.path.exists(INFO_DB_PATH):
            df = pd.read_csv(INFO_DB_PATH)
            match = df[df['Disease'].str.lower() == clean_name]
            if not match.empty:
                row = match.iloc[0]
                tips = [row[f"Precaution_{i}"] for i in range(1, 6) if pd.notna(row.get(f"Precaution_{i}"))]
                return tips, "Local Database"
        return ["Consult a professional for specific advice."], "General"

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
    st.set_page_config(page_title="Advanced AI Health", page_icon="🏥", layout="wide")

    # --- SIDEBAR (Context for the Fragment) ---
    with st.sidebar:
        st.title("📱 Live Vital Monitor")
        render_biometric_content()

    st.title("🏥 AI Health Assistant 2.0")
    st.caption("Monitoring Vitals + Symptom Analysis")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Vitals are syncing in the sidebar. How can I help you?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot = st.session_state.bot
        with st.spinner("Analyzing..."):
            disease, matched, conf = bot.predict(prompt.lower())

            if disease:
                symptoms = bot.get_symptoms(disease)
                advice, src = bot.get_advice(disease)
                response = f"**Assessment:** Potential case of **{disease.upper()}** ({conf:.1f}% match).\n\n"
                response += f"**Symptoms Found:** {', '.join(matched).replace('_', ' ')}\n\n"
                response += f"**Typical Profile:** {', '.join(symptoms[:5])}\n\n"
                response += f"**Advice ({src}):**\n"
                for tip in advice: response += f"- {tip}\n"
            else:
                response = "I couldn't detect specific symptoms. Could you please clarify?"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
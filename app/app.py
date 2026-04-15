import streamlit as st
import os
import pandas as pd
import joblib
import re
import difflib
import time
import warnings
from supabase import create_client, Client  # Added for the bridge

# Suppress the feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# 1. CONFIGURATION
# =======================
# --- SUPABASE BRIDGE CONFIG ---
# Paste your keys from the Supabase Dashboard here
SUPABASE_URL = "https://aetkwxgawqwdkalszpvz.supabase.co"
SUPABASE_KEY = "sb_publishable_1YwAsOGUppsbho7eohAlag_7D43E425"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# ORIGINAL PATHS (Left untouched for your AI logic)
BIO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobile_health_model.pkl")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "disease_and_symptom_clean")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")

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

    # We only check for the MODEL file now, as data comes from the Cloud Bridge
    if os.path.exists(BIO_MODEL_PATH):
        try:
            # 1. Fetch latest record from the Supabase Bridge
            response = supabase.table("vitals").select("*").order("created_at", desc=True).limit(1).execute()

            if response.data:
                latest = response.data[0]
                hr = latest['heart_rate']
                lux = latest['light_level']

                # 2. Load your original local model
                bio_model = joblib.load(BIO_MODEL_PATH)

                # 3. Predict using the Bridge data
                input_df = pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean'])
                prediction = bio_model.predict(input_df)[0]

                # 4. Display Status
                status_icon = "🚨" if prediction == 1 else "✅"
                status_text = "RISK DETECTED" if prediction == 1 else "STABLE"

                status_placeholder.markdown(
                    f"### {status_icon} {status_text}\n"
                    f"**HR:** {hr} BPM | **Lux:** {lux}\n"
                    f"---"
                )
            else:
                status_placeholder.info("Bridge active, waiting for phone data...")
        except Exception as e:
            status_placeholder.warning("Syncing Cloud Bridge...")
    else:
        status_placeholder.info("Model missing. Connect phone to view vitals.")


# ==========================================
# 3. MEDICAL AI CLASS (REMAINS UNTOUCHED)
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
# 4. MAIN UI EXECUTION (REMAINS UNTOUCHED)
# ==========================================
def main():
    st.set_page_config(page_title="Advanced AI Health", page_icon="🏥", layout="wide")

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
        query_lower = prompt.lower().strip()

        with st.spinner("Analyzing..."):
            disease, matched, conf = bot.predict(query_lower)
            if matched:
                response = f"**Assessment:** Potential case of **{disease.upper()}** ({conf:.1f}% match)."
                symptoms = bot.get_symptoms(disease)
                if symptoms: response += f"\n\n**Related symptoms:** {', '.join(symptoms[:5])}"
            else:
                response = "I couldn't detect specific symptoms. Could you please describe them differently?"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
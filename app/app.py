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
LEARNING_DB_PATH = os.path.join(APP_DATA_DIR, "user_feedback_learning.csv")
os.makedirs(APP_DATA_DIR, exist_ok=True)

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection", "cold": "upper respiratory infection",
    "flu": "influenza", "sugar": "diabetes", "bp": "hypertension",
    "heart attack": "myocardial infarction", "brain stroke": "cerebrovascular accident"
}


# ==========================================
# 2. UPDATED BRIDGE LOGIC (Cloud Ingestion)
# ==========================================
@st.fragment(run_every=3)
def render_biometric_content():
    """Sidebar fragment that polls Supabase every 3 seconds"""
    status_placeholder = st.empty()

    if os.path.exists(BIO_MODEL_PATH):
        try:
            # Fetch latest data from Supabase
            response = supabase.table("vitals").select("*").order("created_at", desc=True).limit(1).execute()

            if response.data:
                latest = response.data[0]
                hr, lux = latest['heart_rate'], latest['light_level']

                # Risk Prediction
                bio_model = joblib.load(BIO_MODEL_PATH)
                input_df = pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean'])
                prediction = bio_model.predict(input_df)[0]

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
        status_placeholder.info("Connect phone to view vitals.")


# ==========================================
# 3. ADVANCED MEDICAL AI CLASS
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

    def save_learning_data(self, symptoms, actual_disease):
        """Self-Learning: Saves user-corrected data to CSV"""
        new_data = {"Symptoms": symptoms, "Label": actual_disease, "Time": time.ctime()}
        df = pd.DataFrame([new_data])
        df.to_csv(LEARNING_DB_PATH, mode='a', header=not os.path.exists(LEARNING_DB_PATH), index=False)

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        return [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]

    def scrape_wikipedia(self, disease_name):
        slug = disease_name.strip().replace(" ", "_").title()
        url = f"https://en.wikipedia.org/wiki/{slug}"
        found_data = []
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for p in soup.find_all('p'):
                    if len(p.get_text()) > 50:
                        clean_text = re.sub(r'\[\d+\]', '', p.get_text().strip())
                        found_data.append(f"**Summary:** {clean_text[:300]}...")
                        break
                return found_data, url
        except:
            pass
        return [], "None"

    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        # Check Local DB First
        if os.path.exists(INFO_DB_PATH):
            try:
                df = pd.read_csv(INFO_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    return [row[f"Precaution_{i}"] for i in range(1, 6) if
                            pd.notna(row.get(f"Precaution_{i}"))], "Local"
            except:
                pass

        # Fallback to Scraper
        found_text, source = self.scrape_wikipedia(clean_name)
        return found_text, source

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

    with st.sidebar:
        st.title("📱 Live Vital Monitor")
        render_biometric_content()
        st.divider()
        if st.button("🗑️ Clear Learning Cache"):
            if os.path.exists(LEARNING_DB_PATH): os.remove(LEARNING_DB_PATH)
            st.toast("Learning cache cleared!")

    st.title("🏥 AI Health Assistant 2.0")
    st.caption("Monitoring Vitals + Symptom Analysis")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Enter symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot = st.session_state.bot
        with st.spinner("Analyzing..."):
            disease, matched, conf = bot.predict(prompt.lower())

            if disease:
                symptoms = bot.get_symptoms(disease)
                advice, src = bot.get_advice(disease)

                response = f"**Assessment:** {disease.upper()} ({conf:.1f}% match)\n\n"
                if symptoms: response += f"**🩺 Typical Symptoms:** {', '.join(symptoms[:6])}\n\n"
                response += f"**🛡️ Precautions ({src}):**\n"
                for item in advice: response += f"- {item}\n"

                # Render Feedback for Self-Learning
                with st.chat_message("assistant"):
                    st.markdown(response)
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Correct"): st.toast("Knowledge Confirmed!")
                    if c2.button("❌ Wrong"):
                        bot.save_learning_data(prompt, "Incorrect Prediction")
                        st.toast("Saved for retraining!")
            else:
                response = "I couldn't identify those symptoms. Try 'fever, cough'."
                with st.chat_message("assistant"):
                    st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
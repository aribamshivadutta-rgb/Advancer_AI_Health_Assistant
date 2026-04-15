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

warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# 1. CONFIGURATION
# =======================
SUPABASE_URL = "https://aetkwxgawqwdkalszpvz.supabase.co"
SUPABASE_KEY = "sb_publishable_1YwAsOGUppsbho7eohAlag_7D43E425"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "disease_and_symptom_clean")

BIO_MODEL_PATH = os.path.join(MODEL_DIR, "mobile_health_model.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")

# Paths for Self-Learning & Advice
APP_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "app_data")
LEARNING_DB_PATH = os.path.join(APP_DATA_DIR, "user_feedback_learning.csv")
INFO_DB_PATH = os.path.join(APP_DATA_DIR, "who_data_clean.csv")
os.makedirs(APP_DATA_DIR, exist_ok=True)


# ==========================================
# 2. CLOUD BRIDGE (VITAL MONITOR)
# ==========================================
@st.fragment(run_every=3)
def render_biometric_content():
    status_placeholder = st.empty()
    if os.path.exists(BIO_MODEL_PATH):
        try:
            response = supabase.table("vitals").select("*").order("created_at", desc=True).limit(1).execute()
            if response.data:
                latest = response.data[0]
                hr, lux = latest['heart_rate'], latest['light_level']
                bio_model = joblib.load(BIO_MODEL_PATH)
                prediction = bio_model.predict(pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean']))[0]

                status_icon = "🚨" if prediction == 1 else "✅"
                status_text = "RISK DETECTED" if prediction == 1 else "STABLE"
                status_placeholder.markdown(f"### {status_icon} {status_text}\n**HR:** {hr} BPM | **Lux:** {lux}\n---")
            else:
                status_placeholder.info("Waiting for phone data...")
        except:
            status_placeholder.warning("Syncing Cloud...")
    else:
        status_placeholder.info("Model missing.")


# ==========================================
# 3. MEDICAL AI CLASS (With Scraper & Learning)
# ==========================================
class MedicalAI:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        self.le = joblib.load(LE_PATH) if os.path.exists(LE_PATH) else None
        self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist() if os.path.exists(FEAT_PATH) else []
        self.df_full = pd.read_csv(FULL_DATA_PATH) if os.path.exists(FULL_DATA_PATH) else None

    def scrape_precautions(self, disease_name):
        """Web Scraping Capability: Wikipedia/WHO fallback"""
        slug = disease_name.strip().replace(" ", "_").title()
        url = f"https://en.wikipedia.org/wiki/{slug}"
        precautions = []
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for h in soup.find_all(['h2', 'h3']):
                    if any(t in h.get_text() for t in ["Prevention", "Management", "Treatment"]):
                        ul = h.find_next('ul')
                        if ul: precautions = [li.get_text().strip()[:100] for li in ul.find_all('li')[:3]]
                        break
        except:
            pass
        return precautions if precautions else ["Maintain hygiene", "Consult a doctor", "Monitor symptoms"]

    def save_learning_data(self, symptoms, actual_disease):
        """Self-Learning Capability: Saves corrected data for future training"""
        new_data = {"Symptoms": symptoms, "Correct_Disease": actual_disease, "Timestamp": time.ctime()}
        df = pd.DataFrame([new_data])
        df.to_csv(LEARNING_DB_PATH, mode='a', header=not os.path.exists(LEARNING_DB_PATH), index=False)

    def predict(self, user_input):
        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []
        for t in tokens:
            m = difflib.get_close_matches(t, self.known_symptoms, n=1, cutoff=0.7)
            if m:
                input_dict[m[0]] = 1
                matched.append(m[0])
        if not matched: return None, [], 0
        input_df = pd.DataFrame([input_dict])
        pred_id = self.model.predict(input_df)[0]
        conf = self.model.predict_proba(input_df)[0][pred_id] * 100
        return self.le.inverse_transform([pred_id])[0], matched, conf


# ==========================================
# 4. MAIN UI
# ==========================================
def main():
    st.set_page_config(page_title="Advanced AI Health", page_icon="🏥", layout="wide")

    with st.sidebar:
        st.title("📱 Live Vital Monitor")
        render_biometric_content()
        st.divider()
        st.subheader("🧠 Learning Mode")
        if st.button("Clear Learning Cache"):
            if os.path.exists(LEARNING_DB_PATH): os.remove(LEARNING_DB_PATH)
            st.success("Cache Cleared")

    st.title("🏥 AI Health Assistant 2.0")

    if 'bot' not in st.session_state: st.session_state.bot = MedicalAI()
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Enter symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot = st.session_state.bot
        disease, matched, conf = bot.predict(prompt)

        if disease:
            precautions = bot.scrape_precautions(disease)
            response = f"**Assessment:** {disease.upper()} ({conf:.1f}% match)\n\n"
            response += "**Precautions:**\n" + "\n".join([f"- {p}" for p in precautions])

            # Show Feedback buttons for Self-Learning
            with st.chat_message("assistant"):
                st.markdown(response)
                col1, col2 = st.columns(2)
                if col1.button("✅ Correct"): st.toast("AI Knowledge Reinforced!")
                if col2.button("❌ Wrong"):
                    bot.save_learning_data(prompt, "User Flagged Incorrect")
                    st.toast("Saved for Retraining")
        else:
            response = "I couldn't detect symptoms. Try 'fever, cough'."
            with st.chat_message("assistant"):
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
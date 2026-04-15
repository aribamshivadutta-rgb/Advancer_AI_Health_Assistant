import streamlit as st
import os
import pandas as pd
import joblib
import re
import difflib
import requests
from bs4 import BeautifulSoup
import time

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
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",
    "flu": "influenza",
    "sugar": "diabetes",
    "bp": "hypertension",
    "heart attack": "myocardial infarction",
    "brain stroke": "cerebrovascular accident"
}


# =======================
# 2. BIOMETRIC SIDEBAR LOGIC
# =======================
def display_biometric_sidebar():
    st.sidebar.title("📱 Live Vital Monitor")
    st.sidebar.markdown("---")

    # Toggle for automatic refreshing
    live_mode = st.sidebar.toggle("Live Mode (Auto-Refresh)", value=True)

    if os.path.exists(BIO_DATA_PATH) and os.path.exists(BIO_MODEL_PATH):
        try:
            df_bio = pd.read_csv(BIO_DATA_PATH)
            bio_model = joblib.load(BIO_MODEL_PATH)

            if not df_bio.empty:
                latest = df_bio.iloc[-1]
                hr = latest['HeartRate_Clean']
                lux = latest['Light_Clean']

                # Predict Risk using the Biometric Model
                prediction = bio_model.predict([[hr, lux]])[0]

                # UI Styling
                if prediction == 1:
                    st.sidebar.error("🚨 CRITICAL RISK DETECTED")
                    st.sidebar.metric("Heart Rate", f"{hr} BPM", delta="High", delta_color="inverse")
                else:
                    st.sidebar.success("✅ Vitals Stable")
                    st.sidebar.metric("Heart Rate", f"{hr} BPM")

                st.sidebar.metric("Environment Light", f"{lux} Lux")

                # Heart Rate History Chart
                st.sidebar.subheader("HR History")
                st.sidebar.line_chart(df_bio['HeartRate_Clean'].tail(20))

                st.sidebar.caption(f"Last Sync: {time.strftime('%H:%M:%S')}")

                # Auto-refresh logic
                if live_mode:
                    time.sleep(3)
                    st.rerun()
        except Exception as e:
            st.sidebar.warning("Waiting for mobile stream...")
    else:
        st.sidebar.info("Sync phone data to see live vitals.")


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
                st.error(f"Error loading resources: {e}")
        else:
            st.error("⚠️ Disease Model not found.")

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        active_symptoms = [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]
        return active_symptoms

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
                for h in soup.find_all(['h2', 'h3']):
                    if any(t in h.get_text() for t in ["Prevention", "Management", "Treatment"]):
                        ul = h.find_next('ul')
                        if ul:
                            for li in ul.find_all('li')[:3]:
                                found_data.append(re.sub(r'\[\d+\]', '', li.get_text().strip()))
                        break
                return found_data, url
        except:
            pass
        return [], "None"

    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        if os.path.exists(INFO_DB_PATH):
            try:
                df = pd.read_csv(INFO_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    tips = [row[f"Precaution_{i}"] for i in range(1, 6) if pd.notna(row.get(f"Precaution_{i}"))]
                    return tips, row.get('Source', 'Local')
            except:
                pass

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
        disease = self.le.inverse_transform([pred_id])[0]
        return disease, matched, conf


# =======================
# 4. MAIN UI EXECUTION
# =======================
def main():
    # Use wide layout to accommodate the sidebar and chat comfortably
    st.set_page_config(page_title="Advanced AI Health", page_icon="🏥", layout="wide")

    # Launch Sidebar Monitor
    display_biometric_sidebar()

    # Chat UI Title
    st.title("🏥 AI Health Assistant 2.0")
    st.caption("Monitoring Vitals + Symptom Analysis")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am monitoring your vitals in the sidebar. You can also describe any symptoms you are feeling here."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        bot = st.session_state.bot
        query_lower = prompt.lower().strip()
        search_term = DISEASE_ALIASES.get(query_lower, query_lower)

        response_text = ""
        disease_found = None

        matches = difflib.get_close_matches(search_term, bot.known_diseases, n=1, cutoff=0.85)
        if matches:
            disease_found = matches[0]
            response_text = f"✅ **Found info for:** **{disease_found.title()}**.\n"
        else:
            disease, matched, conf = bot.predict(query_lower)
            if matched:
                response_text = f"**Suspicion:** **{disease.upper()}** ({conf:.1f}% confidence).\n"
                disease_found = disease
            else:
                response_text = "❌ Could not identify symptoms."

        if disease_found:
            symptoms = bot.get_symptoms(disease_found)
            if symptoms:
                response_text += f"\n**🩺 Other Symptoms:** {', '.join(symptoms[:5])}\n"
            advice, src = bot.get_advice(disease_found)
            if advice:
                response_text += f"\n---\n**🛡️ Advice ({src}):**\n"
                for item in advice[:3]: response_text += f"- {item}\n"

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()


if __name__ == "__main__":
    main()
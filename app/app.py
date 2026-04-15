import streamlit as st
import os
import pandas as pd
import joblib
import re
import difflib
import requests
from bs4 import BeautifulSoup

# =======================
# 1. CONFIGURATION (PORTABLE PATHS)
# =======================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "disease_and_symptom_clean")

# Specific File Paths
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
# New: Path to full dataset for symptom lookup
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
# 2. BACKEND LOGIC CLASS
# =======================
class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.df_full = None  # Store full dataset
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.le = joblib.load(LE_PATH)
                self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
                self.known_diseases = [d.lower() for d in self.le.classes_]

                # Load Full Dataset for Symptom Lookup
                if os.path.exists(FULL_DATA_PATH):
                    self.df_full = pd.read_csv(FULL_DATA_PATH)

            except Exception as e:
                st.error(f"Error loading model files: {e}")
        else:
            st.error(f"⚠️ Model not found at: {MODEL_PATH}\nPlease verify your folder structure.")

    # --- GET SYMPTOMS (Reverse Lookup) ---
    def get_symptoms(self, disease_name):
        """Finds all symptoms associated with a specific disease in the dataset."""
        if self.df_full is None: return []

        # Filter for the disease (case-insensitive)
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]

        if subset.empty: return []

        # Get the first matching row
        row = subset.iloc[0]

        # Find columns where value is 1 (Active Symptoms)
        # We check if col is in known_symptoms to avoid 'prognosis' column or indices
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

        found_text = []
        source = "WHO"
        slug = clean_name.replace(" ", "-")
        try:
            resp = requests.get(f"https://www.who.int/news-room/fact-sheets/detail/{slug}",
                                headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for h in soup.find_all(['h2', 'h3']):
                    if any(k in h.get_text() for k in ["Prevention", "Treatment", "Key facts"]):
                        for tag in h.find_next_siblings(['p', 'ul'])[:4]:
                            txt = tag.get_text().strip().replace('\n', ' ')
                            if len(txt) > 20: found_text.append(txt)
                        if found_text: break
        except:
            pass

        if not found_text:
            found_text, source = self.scrape_wikipedia(clean_name)

        if found_text:
            new_row = {"Disease": clean_name, "Source": source}
            for i, tip in enumerate(found_text[:5]): new_row[f"Precaution_{i + 1}"] = tip
            if os.path.exists(INFO_DB_PATH):
                df = pd.read_csv(INFO_DB_PATH)
            else:
                df = pd.DataFrame(
                    columns=["Disease", "Source", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4",
                             "Precaution_5"])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(INFO_DB_PATH, index=False)

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
# 3. STREAMLIT CHAT UI
# =======================
def main():
    st.set_page_config(page_title="Medical AI Chat", page_icon="💬", layout="centered")
    st.title("💬 AI Health Assistant")
    st.caption("Describe your symptoms (e.g., 'fever, headache') or ask about a disease.")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Health Assistant. How can I help you today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your symptoms here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        bot = st.session_state.bot
        query_lower = prompt.lower().strip()
        search_term = DISEASE_ALIASES.get(query_lower, query_lower)

        response_text = ""
        disease_found = None

        matches = difflib.get_close_matches(search_term, bot.known_diseases, n=1, cutoff=0.85)
        if not matches:
            matches = [d for d in bot.known_diseases if search_term in d]

        if matches:
            disease_found = matches[0]
            response_text = f"✅ **Identification:** I found information for **{disease_found.title()}**.\n"
        else:
            disease, matched, conf = bot.predict(query_lower)
            if matched:
                response_text = f"**Based on symptoms** ({', '.join(matched).replace('_', ' ')}), I suspect **{disease.upper()}** (Confidence: {conf:.1f}%).\n"
                disease_found = disease
            else:
                response_text = "❌ I couldn't recognize those symptoms. Please try using standard medical terms."

        if disease_found:
            # 1. Get Symptoms
            symptoms = bot.get_symptoms(disease_found)
            if symptoms:
                response_text += f"\n\n**🩺 Typical Symptoms:**\n"
                response_text += f"Common indications include: {', '.join(symptoms[:8])}.\n"
                response_text += f""

            # 2. Get Advice
            advice, source = bot.get_advice(disease_found)
            response_text += f"\n\n--- \n**🛡️ Recommended Advice** *(Source: {source})*:\n"
            if advice:
                for item in advice:
                    response_text += f"- {item}\n"
            else:
                response_text += "- No specific online advice found."

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()


if __name__ == "__main__":
    main()
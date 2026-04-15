import os
import pandas as pd
import joblib
import re
import difflib
import sys
import requests
from bs4 import BeautifulSoup
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# =======================
# 1. CONFIGURATION
# =======================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "clean", "disease_and_symptom_clean")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
INFO_DB_PATH = os.path.join(DATA_DIR, "who_data_clean.csv")

sys.path.append(SCRIPTS_DIR)

# 🧠 ALIAS DICTIONARY: Translates "Common" names to "Dataset" names
DISEASE_ALIASES = {
    "tb": "tuberculosis",
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",
    "flu": "influenza",
    "piles": "hemorrhoids",
    "sugar": "diabetes",
    "heart attack": "myocardial infarction",
    "brain stroke": "cerebrovascular accident"
}


class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.load_resources()

    def load_resources(self):
        print("\n--- Initializing Medical AI ---")
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model missing at {MODEL_PATH}")
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            self.le = joblib.load(LE_PATH)
            self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
            self.known_diseases = [d.lower() for d in self.le.classes_]
            print(f" -> Model loaded. Knows {len(self.known_diseases)} diseases.")
        except Exception as e:
            print(f"[ERROR] Failed to load AI brain: {e}")

    # =========================================
    # 🌍 WEB SCRAPING LOGIC
    # =========================================

    def scrape_wikipedia(self, disease_name):
        """Backup Scraper: Reads Paragraphs AND Lists from Wikipedia."""
        print(f"   (📖 WHO failed. Trying Wikipedia for '{disease_name}'...)")

        variations = [
            disease_name.strip().replace(" ", "_").title(),
            disease_name.strip().replace(" ", "_").title() + "_(disease)"
        ]

        found_data = []
        target_headers = ["Prevention", "Treatment", "Management", "Mitigation", "Therapy", "Self-care"]

        try:
            for slug in variations:
                url = f"https://en.wikipedia.org/wiki/{slug}"
                resp = requests.get(url, timeout=5)

                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')

                    # 1. Get Summary
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        clean_text = re.sub(r'\[.*?\]', '', text)
                        if len(clean_text) > 50 and "refer to" not in clean_text:
                            found_data.append(f"SUMMARY: {clean_text[:250]}...")
                            break

                    # 2. Get Advice
                    for h in soup.find_all(['h2', 'h3']):
                        header_text = h.get_text().strip()
                        if any(t in header_text for t in target_headers):
                            curr_elem = h.find_next_sibling()
                            while curr_elem and curr_elem.name not in ['h2', 'h3']:
                                if curr_elem.name == 'p':
                                    text = re.sub(r'\[.*?\]', '', curr_elem.get_text().strip())
                                    if len(text) > 40: found_data.append(text)
                                elif curr_elem.name == 'ul':
                                    for li in curr_elem.find_all('li'):
                                        text = re.sub(r'\[.*?\]', '', li.get_text().strip())
                                        if len(text) > 10: found_data.append(text)
                                curr_elem = curr_elem.find_next_sibling()
                                if len(found_data) > 6: break

                    if found_data: return found_data, url

        except Exception as e:
            print(f"   (Wiki Error: {e})")

        return [], "None"

    def get_disease_info(self, disease_name):
        precautions = []
        clean_name = disease_name.lower().strip()

        # 1. CHECK LOCAL FILE
        if os.path.exists(INFO_DB_PATH):
            try:
                df = pd.read_csv(INFO_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    for i in range(1, 6):
                        col = f"Precaution_{i}"
                        if col in row and pd.notna(row[col]):
                            precautions.append(row[col])
                    return precautions, row.get('Source', 'Local')
            except:
                pass

        # 2. TRY WHO SCRAPER
        print(f"   (🌍 Fetching info for '{disease_name}'...)")
        found_text = []
        source = "WHO"

        slug = clean_name.replace(" ", "-")
        urls = [
            f"https://www.who.int/news-room/fact-sheets/detail/{slug}",
            f"https://www.who.int/news-room/fact-sheets/detail/{slug}-disease"
        ]
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            for url in urls:
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        keywords = ["Prevention", "Treatment", "Key facts"]
                        for h in soup.find_all(['h2', 'h3']):
                            if any(k in h.get_text() for k in keywords):
                                siblings = h.find_next_siblings(['p', 'ul'])
                                for tag in siblings[:4]:
                                    txt = tag.get_text().strip().replace('\n', ' ')
                                    if len(txt) > 20: found_text.append(txt)
                                if found_text: break
                        if found_text:
                            source = url
                            break
                except:
                    continue
        except:
            pass

        # 3. TRY WIKIPEDIA (If WHO failed)
        if not found_text:
            found_text, source = self.scrape_wikipedia(clean_name)

        # 4. SAVE RESULT
        if found_text:
            new_row = {"Disease": clean_name, "Source": source}
            for i, tip in enumerate(found_text[:5]):
                new_row[f"Precaution_{i + 1}"] = tip

            if os.path.exists(INFO_DB_PATH):
                df = pd.read_csv(INFO_DB_PATH)
            else:
                df = pd.DataFrame(
                    columns=["Disease", "Source", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4",
                             "Precaution_5"])

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            os.makedirs(os.path.dirname(INFO_DB_PATH), exist_ok=True)
            df.to_csv(INFO_DB_PATH, index=False)

        return found_text, source

    def smart_parse(self, user_input):
        cleaned_input = re.sub(r'\b(and|or|I have|feeling|my|is|very|severe)\b', '', user_input, flags=re.IGNORECASE)
        raw_tokens = [s.strip().replace(" ", "_").lower() for s in cleaned_input.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched_log = []
        for token in raw_tokens:
            if not token: continue
            match = None
            if token in self.known_symptoms:
                match = token
            else:
                matches = difflib.get_close_matches(token, self.known_symptoms, n=1, cutoff=0.7)
                if matches: match = matches[0]
            if match:
                input_dict[match] = 1
                matched_log.append(match)
        return pd.DataFrame([input_dict]), matched_log

    # --- MAIN CHAT ---
    def start_chat(self):
        if not self.model: return
        print("\n" + "=" * 50)
        print(" 🏥 MEDICAL AI ASSISTANT (WHO + Wiki Enabled)")
        print(" (Note: Information is sourced from WHO & Wikipedia.")
        print("  Educational use only. Consult a doctor.)")
        print(" Type symptoms (e.g. 'fever, rash') or a disease name.")
        print("=" * 50)

        while True:
            user_input = input("\nPatient: ").strip().lower()
            if user_input in ['quit', 'exit']: break
            if not user_input: continue

            if user_input == 'list':
                print(f"\n🧠 Known Diseases: {', '.join(self.known_diseases[:15])}...")
                continue

            # 1. CHECK ALIASES
            search_term = DISEASE_ALIASES.get(user_input, user_input)

            # 2. DISEASE NAME LOOKUP
            matches = difflib.get_close_matches(search_term, self.known_diseases, n=1, cutoff=0.85)
            if not matches:
                matches = [d for d in self.known_diseases if search_term in d]

            if matches:
                disease = matches[0]
                print(f"Bot: Identifying '{disease.title()}'...")
                self.show_details(disease)
                continue

            # 3. PREDICT FROM SYMPTOMS
            input_df, matched = self.smart_parse(user_input)
            if not matched:
                print("Bot: I didn't recognize that input. Try 'list' or use simpler words.")
                continue

            pred_id = self.model.predict(input_df)[0]
            confidence = self.model.predict_proba(input_df)[0][pred_id] * 100
            disease = self.le.inverse_transform([pred_id])[0]

            print(f"Bot: Based on {matched}, I suspect: {disease.upper()}")
            print(f"     (Confidence: {confidence:.1f}%)")
            self.show_details(disease)

    def show_details(self, disease):
        advice, source = self.get_disease_info(disease)
        print("-" * 40)

        # --- IF REAL ADVICE IS FOUND ---
        if advice:
            print(f"🛡️ ADVICE FOR {disease.upper()} (Source: {source}):")
            for item in advice:
                # Clean up references like [1]
                item = re.sub(r'\[.*?\]', '', item).strip()
                # Split long paragraphs if needed
                if len(item) > 150:
                    sentences = re.split(r'(?<=[.!?])\s+', item)
                    for s in sentences:
                        if len(s.strip()) > 5:
                            print(f"   • {s.strip()}")
                else:
                    print(f"   • {item}")

        # --- IF NO ADVICE IS FOUND (SAFETY NET) ---
        else:
            print(f"⚠️ IMPORTANT NOTE FOR {disease.upper()}:")
            print("   (Specific online data could not be fetched)")
            print("")
            print("   Please follow these general safety steps:")
            print("   1. 🚑 CONSULT A DOCTOR at the earliest possible.")
            print("   2. Stay hydrated and rest.")
            print("   3. Do not self-medicate without professional advice.")
            print("   4. Monitor your body temperature and symptoms.")

        print("-" * 40)


if __name__ == "__main__":
    bot = MedicalAI()
    bot.start_chat()
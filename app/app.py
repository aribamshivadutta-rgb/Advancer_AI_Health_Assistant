import streamlit as st
import pandas as pd
import joblib
import os
import time
from supabase import create_client, Client

# --- 1. SUPABASE CONFIGURATION ---
# Replace with your actual credentials from the Supabase Dashboard
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. LOAD AI MODELS ---
# Using the relative paths you established for your 4th-sem project
MODEL_PATH = "models/lgbm_model_clean.pkl"
bio_model = joblib.load(MODEL_PATH)


# --- 3. CLOUD BIOMETRIC FRAGMENT ---
@st.fragment(run_every=3)  # Refreshes every 3 seconds for a "live pulse"
def render_biometric_content():
    status_placeholder = st.empty()

    try:
        # Pull the absolute LATEST row from the cloud database
        response = supabase.table("vitals").select("*").order("created_at", desc=True).limit(1).execute()

        if response.data:
            latest = response.data[0]
            hr = latest['heart_rate']
            lux = latest['light_level']

            # Prepare data for LightGBM
            # Ensure column names match what the model was trained on
            input_df = pd.DataFrame([[hr, lux]], columns=['HeartRate_Clean', 'Light_Clean'])
            prediction = bio_model.predict(input_df)[0]

            # Display Status
            if prediction == 1:
                st.toast("🚨 EMERGENCY DETECTED!", icon="⚠️")
                status_msg = "🚨 **EMERGENCY: ABNORMAL VITALS**"
            else:
                status_msg = "✅ **STABLE: MONITORING ACTIVE**"

            status_placeholder.markdown(
                f"### {status_msg}\n---\n**Heart Rate:** {hr} BPM  \n**Light Level:** {lux} Lux")

        else:
            status_placeholder.info("Waiting for phone to sync...")

    except Exception as e:
        status_placeholder.error(f"Cloud Disconnected: Ensure your bridge script is running.")


# --- 4. MAIN UI ---
def main():
    st.set_page_config(page_title="AI Health Assistance 2.0", page_icon="🏥")

    with st.sidebar:
        st.title("📱 Real-time Vitals")
        render_biometric_content()

    st.title("🏥 Advanced AI Health Assistance 2.0")
    st.write("Welcome to your MCA 4th Semester Project Demo.")

    # Your Chat/Symptom logic remains here...
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == "__main__":
    main()
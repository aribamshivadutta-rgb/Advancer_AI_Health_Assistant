import joblib
import pandas as pd
import os
import time
import subprocess
import sys

# Dynamic Path Discovery
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)

# Data & Model Paths
RAW_DATA = os.path.join(BASE_DIR, "data", "temp", "raw_mobile_logs.csv")
CLEAN_DATA = os.path.join(BASE_DIR, "data", "clean", "user_body_input_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobile_health_model.pkl")


def trigger_full_recovery():
    """Automated Orchestration: Repairs the data pipeline if files are missing."""
    print("\n🛠️  SELF-HEALING: Repairing Data & Model...")

    preprocess_path = os.path.join(SCRIPTS_DIR, "preprocessor_bridge.py")
    train_path = os.path.join(SCRIPTS_DIR, "ml_bridge.py")

    # Run Preprocessor
    if os.path.exists(preprocess_path):
        subprocess.run([sys.executable, preprocess_path])

    # Run Training
    if os.path.exists(train_path):
        subprocess.run([sys.executable, train_path])

    print("✅ System Restored.\n")


def run_inference_bridge():
    print("🧠 Smart Inference Monitor: ACTIVE")

    while True:
        # 1. Check for Model & Data
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLEAN_DATA):
            if os.path.exists(RAW_DATA):
                trigger_full_recovery()
            else:
                print("❌ Waiting for Mobile Sync (No Raw Data Found)...")
                time.sleep(10)
                continue

        # 2. Smart Prediction Logic
        try:
            model = joblib.load(MODEL_PATH)
            df = pd.read_csv(CLEAN_DATA)

            if not df.empty:
                # Use both sensors for Context-Aware Prediction
                latest = df.iloc[[-1]][['HeartRate_Clean', 'Light_Clean']]
                prediction = model.predict(latest)

                hr = latest['HeartRate_Clean'].values[0]
                lux = latest['Light_Clean'].values[0]

                status = "🚨 CRITICAL RISK" if prediction[0] == 1 else "✅ STABLE"
                print(f"[{time.strftime('%H:%M:%S')}] {status} | HR: {hr} | Lux: {lux}")

        except Exception as e:
            print(f"⚠️ Monitoring Error: {e}")

        time.sleep(5)


if __name__ == "__main__":
    run_inference_bridge()
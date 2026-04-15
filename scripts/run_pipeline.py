import os
import sys
import subprocess
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import time

# =======================
# CONFIGURATION
# =======================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Sub-Scripts (Now all in the same folder)
CHATBOT_SCRIPT = os.path.join(SCRIPTS_DIR, "run_chatbot.py")
PREPROCESS_SCRIPT = os.path.join(SCRIPTS_DIR, "preprocessor_bridge.py")
TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "ml_bridge.py")
INFERENCE_SCRIPT = os.path.join(SCRIPTS_DIR, "inference_bridge.py")

# [Note: I am omitting the fetch_and_save_advice logic here to keep the code concise,
# but keep your existing versions of those functions in your file!]

def start_mobile_monitoring():
    """Starts the Mobile Inference Bridge in the background."""
    if os.path.exists(INFERENCE_SCRIPT):
        print("\n📱 [SYSTEM] Starting Smart Mobile Monitor in background...")
        # Popen allows the bridge to run while the Chatbot is open
        subprocess.Popen([sys.executable, INFERENCE_SCRIPT], cwd=SCRIPTS_DIR)
    else:
        print(f"⚠️ Warning: {INFERENCE_SCRIPT} not found.")

def main():
    print("\n" + "#" * 60)
    print(" 🚀 ADVANCED AI HEALTH ASSISTANT: SYSTEM STARTUP")
    print("#" * 60)

    # 1. Start Mobile Monitor first so it's ready
    start_mobile_monitoring()

    # 2. Launch Chatbot Interface
    print("\n" + "=" * 60)
    print(" 🏥 LAUNCHING CHATBOT INTERFACE...")
    print("=" * 60)

    if os.path.exists(CHATBOT_SCRIPT):
        subprocess.run([sys.executable, CHATBOT_SCRIPT], cwd=SCRIPTS_DIR)
        print("\n[SYSTEM] Session ended. Shutting down...")
    else:
        print(f"❌ Error: {CHATBOT_SCRIPT} not found.")

if __name__ == "__main__":
    main()
from fastapi import FastAPI
import pandas as pd
import os
from datetime import datetime

app = FastAPI()

# --- FINAL PATH CONFIGURATION ---
# Path updated to your specific requirement
STAGING_DIR = r"E:\Advanced Ai Health Assistant\data\temp"
STAGING_CSV = os.path.join(STAGING_DIR, "raw_mobile_logs.csv")

# Ensure the data\temp directory exists
if not os.path.exists(STAGING_DIR):
    os.makedirs(STAGING_DIR)


@app.get("/")
def health_check():
    return {"status": "Entry Bridge Active", "path": STAGING_DIR}


@app.post("/update")
async def entry_bridge(data: dict):
    """
    STAGE 1: ENTRY BRIDGE
    Receives Android telemetry and stores it in the 'Bronze' layer (raw data).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Capture the raw payload
    raw_payload = {
        "timestamp": timestamp,
        "heart_rate": data.get("heart_rate", 0),
        "light": data.get("light", 0),
        "fall_detected": data.get("fall_detected", False)
    }

    # 2. Commit to raw_mobile_logs.csv in the new path
    df = pd.DataFrame([raw_payload])
    file_exists = os.path.isfile(STAGING_CSV)
    df.to_csv(STAGING_CSV, mode='a', index=False, header=not file_exists)

    # Console feedback for PyCharm
    print(f"📥 [ENTRY] Stored raw log in data\\temp | HR: {raw_payload['heart_rate']}")

    return {"status": "Success", "bridge": "Entry", "location": "data/temp"}
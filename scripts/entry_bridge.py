from fastapi import FastAPI, BackgroundTasks, HTTPException
from supabase import create_client, Client
import pandas as pd
import os
from datetime import datetime

app = FastAPI()

# --- CONFIGURATION ---
STAGING_DIR = r"E:\Advanced Ai Health Assistant\data\temp"
STAGING_CSV = os.path.join(STAGING_DIR, "raw_mobile_logs.csv")

if not os.path.exists(STAGING_DIR):
    os.makedirs(STAGING_DIR)

# --- SUPABASE CONFIG ---
# PASTE YOUR ACTUAL KEYS HERE FROM SETTINGS -> API
SUPABASE_URL = "https://aetkwxgawqwdkalszpvz.supabase.co"
SUPABASE_KEY = "sb_publishable_1YwAsOGUppsbho7eohAlag_7D43E425"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# --- CLOUD SYNC LOGIC ---
def sync_to_supabase(payload: dict):
    """
    Background worker to push biometrics to the Gold Layer (Cloud).
    """
    try:
        cloud_data = {
            "heart_rate": payload["heart_rate"],
            "light_level": payload["light"],  # Mapping 'light' (phone) to 'light_level' (SQL)
            "risk_status": 1 if payload.get("fall_detected") else 0
        }

        # Insert into Supabase 'vitals' table
        supabase.table("vitals").insert(cloud_data).execute()
        print(f"☁️ [CLOUD] Success | HR: {cloud_data['heart_rate']} | Lux: {cloud_data['light_level']}")

    except Exception as e:
        # If internet is down in Imphal, local save still works!
        print(f"📡 [CLOUD] Sync Failed: {e}")


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "project": "Advanced AI Health Assistance 2.0",
        "storage": STAGING_DIR
    }


@app.post("/update")
async def entry_bridge(data: dict, background_tasks: BackgroundTasks):
    """
    ENTRY BRIDGE (Hybrid Persistence Strategy)
    - Records locally for MCA Research (E: Drive)
    - Pushes to Cloud for Live Monitoring (Supabase)
    """
    # 1. Validation: Ensure we aren't getting empty data
    if "heart_rate" not in data:
        raise HTTPException(status_code=400, detail="Missing heart_rate data")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Capture the payload
    raw_payload = {
        "timestamp": timestamp,
        "heart_rate": data.get("heart_rate", 0),
        "light": data.get("light", 0),
        "fall_detected": data.get("fall_detected", False)
    }

    # 3. LOCAL SAVE (Bronze Layer)
    try:
        df = pd.DataFrame([raw_payload])
        file_exists = os.path.isfile(STAGING_CSV)
        df.to_csv(STAGING_CSV, mode='a', index=False, header=not file_exists)
        print(f"📥 [LOCAL] Saved to E:\...temp | HR: {raw_payload['heart_rate']}")
    except Exception as e:
        print(f"❌ [LOCAL] File Error: {e}")

    # 4. CLOUD SYNC (Initiate Background Task)
    background_tasks.add_task(sync_to_supabase, raw_payload)

    return {
        "status": "Accepted",
        "timestamp": timestamp,
        "sync": "Queued"
    }
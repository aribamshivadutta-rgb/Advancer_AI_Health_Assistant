import pandas as pd
import os
import sys


# --- 1. INTERNAL CLEANING LOGIC ---
# We define the function here so we don't need to import anything external
def clean_mobile_data(raw_data):
    """
    Scrubbing logic for heart rate and light sensors.
    Matches the logic used in your Advanced AI Health Assistant 2.0.
    """
    hr = raw_data.get("heart_rate", 0)
    light = raw_data.get("light", 0)

    # Example Cleaning: Remove glitchy 0 BPM readings
    clean_hr = hr if hr > 30 else 72  # Default to 72 if sensor glitch
    clean_light = light if light >= 0 else 0

    return [clean_hr, clean_light]


# --- 2. THE BRIDGE LOGIC ---
def run_preprocessor_bridge():
    # Corrected Paths for your specific E: drive structure
    SOURCE_PATH = r"E:\Advanced Ai Health Assistant\data\temp\raw_mobile_logs.csv"
    DEST_DIR = r"E:\Advanced Ai Health Assistant\data\clean"
    DEST_FILE = os.path.join(DEST_DIR, "user_body_input_clean.csv")

    # Path Verification
    if not os.path.exists(SOURCE_PATH):
        print(f"❌ Data Error: I cannot see the raw file at: {SOURCE_PATH}")
        print("💡 Hint: Ensure the Entry Bridge is running and you have synced your phone.")
        return

    print(f"📂 Data Found! Processing logs from {SOURCE_PATH}...")

    # --- 3. TRANSFORMATION ---
    try:
        raw_df = pd.read_csv(SOURCE_PATH)
        processed_records = []

        for _, row in raw_df.iterrows():
            # Calling the internal function directly
            features = clean_mobile_data({
                "heart_rate": row['heart_rate'],
                "light": row['light']
            })
            processed_records.append(features)

        # Ensure the 'clean' folder exists
        if not os.path.exists(DEST_DIR):
            os.makedirs(DEST_DIR)

        # Save the "Gold" dataset
        df_clean = pd.DataFrame(processed_records, columns=['HeartRate_Clean', 'Light_Clean'])
        df_clean.to_csv(DEST_FILE, index=False)

        print(f"✅ Success! Clean data saved to: {DEST_FILE}")

    except Exception as e:
        print(f"❌ Processing Error: {e}")


if __name__ == "__main__":
    run_preprocessor_bridge()
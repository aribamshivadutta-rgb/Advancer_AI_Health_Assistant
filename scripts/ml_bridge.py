import joblib
import pandas as pd
import os
import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_ml_bridge():
    # Use Relative Paths so it works on any computer
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "user_body_input_clean.csv")
    MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "mobile_health_model.pkl")

    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"❌ Data not found at {CLEAN_DATA_PATH}")
        return

    # 1. Load Data
    df = pd.read_csv(CLEAN_DATA_PATH)
    if len(df) < 10:
        print("❌ Need at least 10 samples to train.")
        return

    # 2. Features and Smart Labels
    X = df[['HeartRate_Clean', 'Light_Clean']]
    y = []
    for _, row in df.iterrows():
        hr = row['HeartRate_Clean']
        lux = row['Light_Clean']
        # Emergency: High HR + Dark (Fall) or Very High HR (Tachycardia)
        if (hr > 125 and lux < 50) or (hr > 160):
            y.append(1)
        else:
            y.append(0)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Training with small-data optimization
    model = LGBMClassifier(n_estimators=50, num_leaves=10, min_child_samples=5)
    model.fit(X_train, y_train)

    # 5. Save the Model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"📦 Smart Model saved/updated at {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    run_ml_bridge()
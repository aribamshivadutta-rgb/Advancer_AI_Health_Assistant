import joblib
import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def run_ml_bridge():
    # Paths
    CLEAN_DATA_PATH = r"E:\Advanced Ai Health Assistant\data\clean\user_body_input_clean.csv"
    MODEL_OUTPUT_PATH = r"E:\Advanced Ai Health Assistant\models\mobile_health_model.pkl"

    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"❌ ML Bridge: Clean data not found at {CLEAN_DATA_PATH}")
        return

    # 1. Load Data
    df = pd.read_csv(CLEAN_DATA_PATH)
    if len(df) < 10:
        print("❌ Not enough data for smart training. Need at least 10 samples!")
        return

    # 2. Features and SMART Labels
    X = df[['HeartRate_Clean', 'Light_Clean']]

    # --- SMART LABELING LOGIC ---
    # Class 0 = Stable | Class 1 = Critical Risk
    y = []
    for _, row in df.iterrows():
        hr = row['HeartRate_Clean']
        lux = row['Light_Clean']

        # Scenario A: Potential Fall/Emergency (High HR + Very Dark/Phone Dropped)
        if hr > 125 and lux < 50:
            y.append(1)
        # Scenario B: Extreme Health Risk (Tachycardia over 160 regardless of light)
        elif hr > 160:
            y.append(1)
        # Scenario C: Active but Safe (High HR in bright light = Exercise)
        else:
            y.append(0)

    # 3. THE SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. TRAINING
    print(f"🚀 Training Smart Model on {len(X_train)} samples...")
    # We use a small number of leaves for LightGBM since our dataset is currently small
    model = LGBMClassifier(n_estimators=50, num_leaves=10, min_child_samples=5)
    model.fit(X_train, y_train)

    # 5. TESTING
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("-" * 30)
    print(f"✅ Smart ML Bridge Results:")
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("-" * 30)

    # 6. Save the Model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"📦 Smart Model saved at {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    run_ml_bridge()
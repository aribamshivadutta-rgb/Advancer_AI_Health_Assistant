import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

# 1. Generate Synthetic Mobile Data for training
# We create 1000 samples of heart rate and light levels
np.random.seed(42)
data_size = 1000

heart_rate = np.random.randint(50, 130, size=data_size)
light_level = np.random.randint(0, 1000, size=data_size)

# Logic: High Heart Rate (>100) or Low Heart Rate (<50) = Health Risk (1)
# High light at night or low light during day could be context,
# but for now, let's keep it simple:
target = ((heart_rate > 100) | (heart_rate < 55)).astype(int)

df_mobile = pd.DataFrame({
    'heart_rate': heart_rate,
    'light': light_level,
    'health_status': target
})

# 2. Train the Mobile-Specific Model
X = df_mobile[['heart_rate', 'light']]
y = df_mobile['health_status']

model = lgb.LGBMClassifier()
model.fit(X, y)

# 3. Save to models folder
if not os.path.exists('../models'):
    os.makedirs('../models')

joblib.dump(model, '../models/mobile_health_model.pkl')
print("Successfully trained and saved: mobile_health_model.pkl")
"""
train_model.py
---------------
Trains a sales prediction model based on advertising data.
Automatically handles file paths, creates required folders,
logs all major steps, and saves both the trained model and scaler.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import custom logger
try:
    from utils.log import log_info, log_error
except ImportError:
    # Fallback if utils not found
    def log_info(msg): print(f"[INFO] {msg}")
    def log_error(msg): print(f"[ERROR] {msg}")

# ===============================
# 1️⃣ PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

DATA_PATH = os.path.join(DATA_DIR, "sales_data.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Create folders if missing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 2️⃣ LOAD DATASET
# ===============================
if not os.path.exists(DATA_PATH):
    log_error(f"❌ Dataset not found at {DATA_PATH}. Please place 'sales_data.csv' in the /data folder.")
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

log_info("📂 Loading dataset...")
data = pd.read_csv(DATA_PATH)
log_info(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# ===============================
# 3️⃣ FEATURE SELECTION
# ===============================
required_columns = ['TV', 'Radio', 'Newspaper', 'Sales']
if not all(col in data.columns for col in required_columns):
    missing = [col for col in required_columns if col not in data.columns]
    log_error(f"❌ Missing required columns: {missing}")
    raise KeyError(f"Missing required columns: {missing}")

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# ===============================
# 4️⃣ TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_info("📊 Dataset split into 80% training and 20% testing sets")

# ===============================
# 5️⃣ FEATURE SCALING
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_info("⚙️ Features scaled successfully using StandardScaler")

# Save processed data (optional)
processed_df = pd.DataFrame(X_train_scaled, columns=['TV', 'Radio', 'Newspaper'])
processed_df['Sales'] = y_train.values
processed_df.to_csv(PROCESSED_PATH, index=False)
log_info(f"💾 Processed training data saved to: {PROCESSED_PATH}")

# ===============================
# 6️⃣ MODEL TRAINING
# ===============================
log_info("🚀 Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
log_info("✅ Model training completed successfully")

# ===============================
# 7️⃣ MODEL EVALUATION
# ===============================
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

log_info("📈 Model Performance Metrics:")
log_info(f"   • MAE : {mae:.3f}")
log_info(f"   • MSE : {mse:.3f}")
log_info(f"   • R²  : {r2:.3f}")

# ===============================
# 8️⃣ SAVE MODEL AND SCALER
# ===============================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

log_info(f"💾 Model saved to: {MODEL_PATH}")
log_info(f"💾 Scaler saved to: {SCALER_PATH}")

# ===============================
# 9️⃣ TEST SAMPLE PREDICTION
# ===============================
sample_input = np.array([[230.1, 37.8, 69.2]])
scaled_sample = scaler.transform(sample_input)
sample_prediction = model.predict(scaled_sample)[0]
log_info(f"🔮 Sample Prediction for [230.1, 37.8, 69.2]: {sample_prediction:.2f}")

print("\n✅ Training complete. Check 'logs/' for detailed records.")

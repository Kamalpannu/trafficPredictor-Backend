import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# -----------------------------
# Data Simulation
# -----------------------------
num_rows = 1000
start_time = datetime.now()
timestamps = [(start_time - timedelta(minutes=i)).timestamp() for i in range(num_rows)]
zones = ['Zone 1', 'Zone 2', 'Zone 3']

pollution = np.random.rand(num_rows)
zone_data = [random.choice(zones) for _ in range(num_rows)]

traffic = []
for i in range(num_rows):
    base = pollution[i] * random.uniform(0.8, 1.2)
    zone_effect = zones.index(zone_data[i]) * 0.5
    noise = random.uniform(-0.1, 0.1)
    traffic.append(base + zone_effect + noise)

df = pd.DataFrame({
    'zone': zone_data,
    'traffic': traffic,
    'pollution': pollution,
    'timestamp': timestamps
})

# -----------------------------
# Features & Labels
# -----------------------------
X = df[['zone', 'pollution']]
y = df['traffic']

encoder = OneHotEncoder(sparse_output=False)
zone_encoded = encoder.fit_transform(X[['zone']])
X_encoded = np.hstack([zone_encoded, X[['pollution']].values])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test, zones_train, zones_test = train_test_split(
    X_encoded, y, df['zone'], test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Overall Evaluation
# -----------------------------
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"Overall R² score: {score:.3f}")
print("Actual traffic values (first 5):", y_test[:5].values)
print("Predicted traffic values (first 5):", y_pred[:5])

# -----------------------------
# Per-Zone Evaluation
# -----------------------------
print("\nPer-Zone Evaluation:")
for zone in zones:
    mask = zones_test == zone
    y_true_zone = y_test[mask]
    y_pred_zone = y_pred[mask]
    if len(y_true_zone) == 0:
        continue
    r2 = r2_score(y_true_zone, y_pred_zone)
    mae = mean_absolute_error(y_true_zone, y_pred_zone)
    mse = mean_squared_error(y_true_zone, y_pred_zone)
    rmse = np.sqrt(mse)
    print(f"{zone} -> R²: {r2:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}")

# -----------------------------
# Save Model & Encoder
# -----------------------------
joblib.dump(model, "model.joblib")
joblib.dump(encoder, "encoder.joblib")

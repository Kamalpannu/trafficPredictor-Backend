from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import random
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from app.database import get_db
from app.models.models import TrafficRecord, PredictionRecord
from app.schemas import TrafficRecordRead, PredictionRecordRead

router = APIRouter()
logger = logging.getLogger("traffic")
ZONES = ["Zone 1", "Zone 2", "Zone 3"]

# ----------------------------
# RandomForest-based Predictor
# ----------------------------
class TrafficPredictorRF:
    """RandomForest-based traffic predictor with per-zone evaluation and minute-by-minute prediction."""
    def __init__(self):
        self.model = None
        self.encoder = None
        self.last_df = None

    def train_model(self, df: pd.DataFrame):
        """Train RandomForest model using zones and traffic_count."""
        if df.empty:
            raise ValueError("No data to train model.")
        self.last_df = df.copy()

        X = df[['zone', 'traffic_count']]
        y = df['traffic_count']

        self.encoder = OneHotEncoder(sparse_output=False)
        zone_encoded = self.encoder.fit_transform(X[['zone']])
        X_encoded = np.hstack([zone_encoded, X[['traffic_count']].values.reshape(-1, 1)])

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_encoded, y)

    def predict_future(self, zone: str, minutes_ahead: int = 30):
        """Predict next `minutes_ahead` traffic counts per zone using iterative predictions."""
        if self.last_df is None or self.model is None:
            raise ValueError("Model not trained yet.")
        df_zone = self.last_df[self.last_df['zone'] == zone]
        if df_zone.empty:
            raise ValueError(f"No data for {zone}")

        last_count = df_zone['traffic_count'].iloc[-1]
        now = pd.Timestamp.now()
        predictions = {}

        # Iteratively predict traffic
        for i in range(1, minutes_ahead + 1):
            # Prepare input for model
            zone_encoded = self.encoder.transform([[zone]])
            X_input = np.hstack([zone_encoded, [[last_count]]])
            pred = self.model.predict(X_input)[0]

            # Save prediction
            future_time = now + timedelta(minutes=i)
            predictions[future_time.isoformat()] = float(pred)

            # Update last_count for next iteration
            last_count = pred

        return predictions

    def evaluate_model(self, zone: str):
        """Evaluate model performance for a zone using training data."""
        if self.last_df is None or self.model is None:
            raise ValueError("Model not trained yet.")
        df_zone = self.last_df[self.last_df['zone'] == zone]
        if df_zone.empty:
            raise ValueError(f"No data for {zone}")

        X = df_zone[['zone', 'traffic_count']]
        y_true = df_zone['traffic_count'].values
        zone_encoded = self.encoder.transform(X[['zone']])
        X_encoded = np.hstack([zone_encoded, X[['traffic_count']].values.reshape(-1, 1)])
        y_pred = self.model.predict(X_encoded)

        metrics = {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }
        return metrics

# ----------------------------
# Router Logic
# ----------------------------
predictor = TrafficPredictorRF()

def simulate_traffic(zone: str, now: datetime) -> int:
    """Simulate traffic count with random spikes."""
    base = random.randint(20, 100)
    if now.hour in [8, 9, 17, 18]:
        base += random.randint(50, 100)
    return base

@router.get("/current-traffic", response_model=list[TrafficRecordRead])
async def get_current_traffic(db: AsyncSession = Depends(get_db)):
    """Simulate and save traffic data, return latest per zone."""
    now = datetime.utcnow()
    records = []
    for zone in ZONES:
        traffic_count = simulate_traffic(zone, now)
        record = TrafficRecord(timestamp=now, zone=zone, traffic_count=traffic_count)
        db.add(record)
        records.append(record)
    await db.commit()

    # Fetch latest per zone
    latest_records = []
    for zone in ZONES:
        stmt = select(TrafficRecord).where(TrafficRecord.zone == zone).order_by(desc(TrafficRecord.timestamp)).limit(1)
        result = await db.execute(stmt)
        latest = result.scalar_one_or_none()
        if latest:
            latest_records.append(latest)
    logger.info("Current traffic records returned.")
    return latest_records

@router.post("/predict-traffic", response_model=list[PredictionRecordRead])
async def predict_traffic(db: AsyncSession = Depends(get_db)):
    """Train/load model, predict next 30min per zone, save and return predictions."""
    stmt = select(TrafficRecord)
    result = await db.execute(stmt)
    traffic_data = result.scalars().all()

    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "zone": r.zone,
        "traffic_count": r.traffic_count
    } for r in traffic_data])
    if df.empty:
        raise HTTPException(status_code=400, detail="No traffic data available for prediction.")

    # Train RandomForest model
    predictor.train_model(df)

    predictions = []
    now = datetime.utcnow()
    for zone in ZONES:
        pred_dict = predictor.predict_future(zone, minutes_ahead=30)
        last_time = list(pred_dict.keys())[-1]
        last_pred = pred_dict[last_time]
        record = PredictionRecord(timestamp=now, zone=zone, predicted_count=int(last_pred))
        db.add(record)
        predictions.append(record)

        # Log per-zone evaluation
        metrics = predictor.evaluate_model(zone)
        logger.info(f"Evaluation for {zone}: {metrics}")

    await db.commit()

    # Return latest predictions per zone
    latest_preds = []
    for zone in ZONES:
        stmt = select(PredictionRecord).where(PredictionRecord.zone == zone).order_by(desc(PredictionRecord.timestamp)).limit(1)
        result = await db.execute(stmt)
        latest = result.scalar_one_or_none()
        if latest:
            latest_preds.append(latest)

    logger.info("Traffic predictions returned.")
    return latest_preds

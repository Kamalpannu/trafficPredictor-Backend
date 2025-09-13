"""
Service to simulate traffic data for each zone, including random spikes for rush hours.
"""
import random
from datetime import datetime, timedelta
from typing import List, Dict

ZONES = ["Zone 1", "Zone 2", "Zone 3"]

async def simulate_current_traffic() -> List[Dict]:
    """Simulate current traffic data for all zones."""
    now = datetime.utcnow().isoformat()
    traffic = []
    for zone in ZONES:
        # Simulate base traffic level
        base = random.randint(20, 80)
        # Add random spike for rush hour
        spike = random.randint(0, 40) if random.random() < 0.3 else 0
        traffic.append({
            "zone": zone,
            "traffic_level": base + spike,
            "timestamp": now
        })
    return traffic

async def simulate_traffic_prediction() -> List[Dict]:
    """Simulate traffic prediction for next 30 minutes for all zones."""
    predictions = []
    now = datetime.utcnow()
    for zone in ZONES:
        pred_levels = []
        pred_times = []
        for i in range(6):  # 5-min intervals for 30 mins
            base = random.randint(20, 80)
            spike = random.randint(0, 40) if random.random() < 0.3 else 0
            pred_levels.append(base + spike)
            pred_times.append((now + timedelta(minutes=5*i)).isoformat())
        predictions.append({
            "zone": zone,
            "predicted_traffic": pred_levels,
            "prediction_timestamps": pred_times
        })
    return predictions

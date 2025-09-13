"""
Router for traffic data and prediction endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.schemas import ZoneTraffic, TrafficPrediction, TrafficResponse
from app.services.traffic_simulation import simulate_current_traffic, simulate_traffic_prediction
import logging

router = APIRouter(prefix="", tags=["Traffic"])
logger = logging.getLogger(__name__)

@router.get("/current-traffic", response_model=list[ZoneTraffic])
async def get_current_traffic():
    """Return current traffic data for all zones."""
    try:
        traffic = await simulate_current_traffic()
        return traffic
    except Exception as e:
        logger.error(f"Error in /current-traffic: {e}")
        raise HTTPException(status_code=500, detail="Failed to get current traffic data.")

@router.get("/predict-traffic", response_model=list[TrafficPrediction])
async def get_traffic_prediction():
    """Predict traffic for next 30 minutes for all zones."""
    try:
        prediction = await simulate_traffic_prediction()
        return prediction
    except Exception as e:
        logger.error(f"Error in /predict-traffic: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict traffic.")

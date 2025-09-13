"""
Pydantic schemas for traffic data and prediction responses.
"""
from pydantic import BaseModel
from typing import List, Dict

class ZoneTraffic(BaseModel):
    zone: str
    traffic_level: int
    timestamp: str

class TrafficPrediction(BaseModel):
    zone: str
    predicted_traffic: List[int]
    prediction_timestamps: List[str]

class TrafficResponse(BaseModel):
    current: List[ZoneTraffic]
    prediction: List[TrafficPrediction]

class RecommendationRequest(BaseModel):
    current: List[ZoneTraffic]
    prediction: List[TrafficPrediction]

class RecommendationResponse(BaseModel):
    recommendations: Dict[str, str]

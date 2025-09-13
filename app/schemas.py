from pydantic import BaseModel
from datetime import datetime

class TrafficRecordCreate(BaseModel):
    timestamp: datetime
    zone: str
    traffic_count: int

class TrafficRecordRead(TrafficRecordCreate):
    id: int

    class Config:
        orm_mode = True

class PredictionRecordCreate(BaseModel):
    timestamp: datetime
    zone: str
    predicted_count: int

class PredictionRecordRead(PredictionRecordCreate):
    id: int

    class Config:
        orm_mode = True
from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base

class TrafficRecord(Base):
    __tablename__ = "traffic_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    zone = Column(String, nullable=False)
    traffic_count = Column(Integer, nullable=False)

class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    zone = Column(String, nullable=False)
    predicted_count = Column(Integer, nullable=False)
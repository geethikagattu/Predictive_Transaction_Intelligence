from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .db import Base

class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String)
    user_id = Column(String)
    risk_score = Column(Float)
    risk_label = Column(String)
    explanation = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from src.database.db import SessionLocal
from src.database.models import RiskPrediction
from src.inference.predict import get_risk_label
from datetime import datetime

app = FastAPI(title="Predictive Transaction Intelligence API")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@app.post("/predict")
def predict(transaction_id: str, user_id: str, risk_score: float, explanation: str, db: Session = Depends(get_db)):
    risk_label = get_risk_label(risk_score)

    record = RiskPrediction(
        transaction_id=transaction_id,
        user_id=user_id,
        risk_score=risk_score,
        risk_label=risk_label,
        explanation=explanation,
        timestamp=datetime.utcnow()
    )

    db.add(record)
    db.commit()

    return {
        "transaction_id": transaction_id,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "explanation": explanation
    }

@app.get("/risk_distribution")
def risk_distribution(db: Session = Depends(get_db)):
    data = db.query(RiskPrediction.risk_label).all()

    distribution = {"Low": 0, "Medium": 0, "High": 0}
    for (label,) in data:
        distribution[label] += 1

    return distribution

@app.get("/explanation_history")
def explanation_history(db: Session = Depends(get_db)):
    records = db.query(RiskPrediction).order_by(RiskPrediction.timestamp.desc()).all()

    return [
        {
            "transaction_id": r.transaction_id,
            "user_id": r.user_id,
            "risk_score": r.risk_score,
            "risk_label": r.risk_label,
            "explanation": r.explanation,
            "timestamp": r.timestamp
        }
        for r in records
    ]

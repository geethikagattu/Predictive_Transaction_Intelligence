from fastapi import APIRouter
from src.database.mysql_connection import get_mysql_connection
import joblib
import pandas as pd
import json

# Create router
router = APIRouter()

# ---------------------------------------------------
# 1. GET SAMPLE DATA FROM MYSQL
# ---------------------------------------------------
@router.get("/fraud-data")
def get_fraud_data():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM transactions LIMIT 100;")
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return {"data": data}

# ---------------------------------------------------
# 2. LOAD TRAINED ML MODEL
# ---------------------------------------------------
model = joblib.load("models/fraud_model.pkl")

# ---------------------------------------------------
# 3. PREDICT ENDPOINT
# ---------------------------------------------------
@router.post("/predict")
def predict_transaction(data: dict):
    # Convert JSON input to DataFrame
    df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }

# ---------------------------------------------------
# 4. METRICS ENDPOINT
# ---------------------------------------------------
@router.get("/metrics")
def get_metrics():
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics

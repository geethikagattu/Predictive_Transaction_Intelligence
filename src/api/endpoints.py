from fastapi import APIRouter
from src.database.mysql_connection import get_mysql_connection
import joblib
import pandas as pd
import json

router = APIRouter()

# ---------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------
model = joblib.load("models/fraud_model.pkl")

# ---------------------------------------------------
# 1. FETCH SAMPLE TRANSACTIONS FROM MYSQL
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
# 2. PREDICT + STORE RISK & EXPLANATION
# ---------------------------------------------------
@router.post("/predict")
def predict_transaction(data: dict):
    df = pd.DataFrame([data])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    # Simple explanation logic (LLM-ready placeholder)
    if probability > 0.7:
        explanation = "High risk due to unusual transaction pattern and high amount"
    elif probability > 0.4:
        explanation = "Moderate risk based on transaction behavior"
    else:
        explanation = "Low risk transaction, behavior matches historical patterns"

    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO risk_predictions
        (transaction_id, fraud_probability, prediction, explanation)
        VALUES (%s, %s, %s, %s)
    """, (
        data.get("transaction_id", "NA"),
        probability,
        prediction,
        explanation
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "prediction": prediction,
        "fraud_probability": probability,
        "explanation": explanation
    }

# ---------------------------------------------------
# 3. RISK DISTRIBUTION ENDPOINT
# ---------------------------------------------------
@router.get("/risk-distribution")
def risk_distribution():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            CASE 
                WHEN fraud_probability > 0.7 THEN 'High'
                WHEN fraud_probability > 0.4 THEN 'Medium'
                ELSE 'Low'
            END AS risk_level,
            COUNT(*) AS count
        FROM risk_predictions
        GROUP BY risk_level;
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data

# ---------------------------------------------------
# 4. EXPLANATION HISTORY ENDPOINT
# ---------------------------------------------------
@router.get("/explanation-history")
def explanation_history():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            transaction_id,
            fraud_probability,
            prediction,
            explanation,
            created_at
        FROM risk_predictions
        ORDER BY created_at DESC
        LIMIT 50;
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data

# ---------------------------------------------------
# 5. MODEL METRICS ENDPOINT
# ---------------------------------------------------
@router.get("/metrics")
def get_metrics():
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics

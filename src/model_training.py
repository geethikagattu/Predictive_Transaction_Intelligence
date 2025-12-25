import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import json
import os

# -----------------------------
# LOAD NEW DATASET
# -----------------------------
CSV_PATH = r"C:\Users\kames\Desktop\new dataset\PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(CSV_PATH)

print("\nDataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -----------------------------
# SELECT FEATURES & TARGET
# -----------------------------
target_col = "isFraud"

feature_cols = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud"
]

X = df[feature_cols].copy()
y = df[target_col].copy()

# -----------------------------
# PREPROCESSING SETUP
# -----------------------------
numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud"
]

categorical_features = ["type"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------------
# MODEL SELECTION
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining model...")
clf.fit(X_train, y_train)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
print("\nEvaluating model...")

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC-ROC:", auc)

# -----------------------------
# SAVE MODEL & METRICS
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/fraud_model.pkl")

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "auc_roc": float(auc),
    "confusion_matrix": cm.tolist()
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel and metrics saved in models/ folder!")

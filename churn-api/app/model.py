# app/model.py
import joblib
import pandas as pd
from pathlib import Path

from preprocessing import (
    encode_binary_columns,
    collapse_service_columns,
    one_hot_encode_multi_category,
    scale_numeric_features,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# loaded once, when this module is first imported — not on every request
_model = joblib.load(MODEL_DIR / "churn_pipeline.pkl")
_scaler = joblib.load(MODEL_DIR / "scaler.pkl")
_feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")


def preprocess_request(customer: dict) -> pd.DataFrame:
    """Take a raw customer dict (matches CustomerRequest schema) and
    run it through the exact same preprocessing steps used in training."""
    df = pd.DataFrame([customer])
    df = encode_binary_columns(df)
    df = collapse_service_columns(df)
    df = one_hot_encode_multi_category(df)
    df, _ = scale_numeric_features(df, scaler=_scaler)

    # a single request won't generate every one-hot column that training
    # data did (e.g. it can only be ONE PaymentMethod, not all four),
    # so we reindex to match the training columns exactly, filling
    # any missing ones with 0
    df = df.reindex(columns=_feature_columns, fill_value=0)
    return df


def predict_churn(customer: dict) -> dict:
    X = preprocess_request(customer)
    probability = _model.predict_proba(X)[0][1]
    prediction = int(probability >= 0.5)

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_level": risk_level,
    }
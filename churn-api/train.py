# train.py
"""
Trains the Northline Mobile churn prediction model.

Run from the project root:
    python train.py
"""

import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).resolve().parent / "app"))
from preprocessing import (
    clean_total_charges,
    encode_binary_columns,
    collapse_service_columns,
    one_hot_encode_multi_category,
    scale_numeric_features,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = clean_total_charges(df)
    df = encode_binary_columns(df)
    df = collapse_service_columns(df)
    df = one_hot_encode_multi_category(df)
    return df


def main():
    print("Loading and preprocessing data...")
    df = load_and_prepare_data()

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Scaling numeric features...")
    X_train, scaler = scale_numeric_features(X_train)
    X_test, _ = scale_numeric_features(X_test, scaler=scaler)

    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\nEvaluation on held-out test set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Saving model artifacts...")
    joblib.dump(model, MODEL_DIR / "churn_pipeline.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.pkl")

    print(f"Done. Artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
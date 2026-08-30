# %%writefile ../app/preprocessing.py
import pandas as pd

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric, filling blanks (new customers, tenure=0) with 0."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


# in app/preprocessing.py, replace encode_binary_columns

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map simple Yes/No and Female/Male columns to 1/0.
    Skips any column not present (e.g. 'Churn' won't exist at inference time)."""
    df = df.copy()
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    return df


def collapse_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 'No internet/phone service' into 'No', then map to 1/0."""
    df = df.copy()
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df


def one_hot_encode_multi_category(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the true multi-category columns (no natural order)."""
    df = df.copy()
    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    return df

    # add to app/preprocessing.py
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(df: pd.DataFrame, scaler: StandardScaler = None):
    """Scale the numeric (non-binary) columns so they're on a comparable range.

    Returns the transformed df and the fitted scaler, so the same scaler
    (fit only on training data) can be reused at inference time.
    """
    df = df.copy()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler
# churn-api

A customer churn prediction model served as a live API. Built for a fictional telecom company, Northline Mobile, as part of my ML deployment series on Towards Data Science, where I'm learning to take machine learning models from a notebook to a real, callable service.

This is the local-first stage of the project: a trained model wrapped in FastAPI, running on your own machine. The next stage (Part B) containerizes it and deploys it to AWS.

<img width="2720" height="2240" alt="churn_api_request_flow" src="https://github.com/user-attachments/assets/98b06cfe-ce86-4b30-82cb-b01feb0d1ed9" />

## What it does

Given a Northline Mobile customer's account details (contract type, tenure, billing info, service add-ons), the API returns a churn probability, a binary prediction, and a simple risk bucket.

```json
{
  "churn_probability": 0.3136,
  "prediction": 0,
  "risk_level": "Medium"
}
```

## Tech stack

- **Python 3** for the model and API
- **pandas / scikit-learn** for preprocessing and training (logistic regression)
- **FastAPI** for the HTTP layer, with **Pydantic** handling request/response validation
- **uvicorn** as the ASGI server
- **joblib** for model serialization

## Project structure

```
churn-api/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_eda.ipynb          # exploratory data analysis
├── app/
│   ├── main.py                 # FastAPI app and routes
│   ├── schemas.py               # request/response models
│   ├── model.py                  # loads artifacts, runs predictions
│   └── preprocessing.py           # shared cleaning/encoding, used by both training and inference
├── models/
│   ├── churn_pipeline.pkl       # trained logistic regression model
│   ├── scaler.pkl                 # fitted StandardScaler
│   └── feature_columns.pkl         # exact column order the model expects
├── train.py                    # reproducible training script
└── requirements.txt
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle), 7,043 customer records with 21 raw fields.

## Model

Baseline logistic regression, trained on an 80/20 stratified split. Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled with `StandardScaler`.

| Metric | Class 0 (stayed) | Class 1 (churned) |
|---|---|---|
| Precision | 0.85 | 0.66 |
| Recall | 0.90 | 0.56 |
| F1-score | 0.87 | 0.60 |

**Overall accuracy: 81%**

This is a baseline, not a tuned model. The focus of this stage of the project is deployment, not model performance. Churn recall (56%) is a known limitation worth improving in a future pass, likely with class-weighting or a different algorithm.

## Setup

```bash
git clone <your-repo-url>
cd churn-api
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This reads the dataset, runs it through preprocessing, trains the model, evaluates it on a held-out test set, and saves three artifacts to `models/`.

## Run the API

```bash
cd app
uvicorn main:app --reload
```

The API will be running at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

## API reference

### `GET /`

Health check.

```json
{ "status": "ok", "service": "churn-api" }
```

### `POST /predict`

Returns a churn prediction for a single customer.

**Request body:**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 890.5
}
```

**Response:**

```json
{
  "churn_probability": 0.3136,
  "prediction": 0,
  "risk_level": "Medium"
}
```

Invalid or missing fields return a `422` with details on exactly what failed, before the request ever reaches the model.

**Example with curl:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 75.5, "TotalCharges": 890.5
  }'
```

## Known limitations

- Runs locally only, not yet reachable outside your own machine
- Baseline model, not tuned for churn recall
- `risk_level` thresholds (Low < 0.3, Medium 0.3-0.6, High > 0.6) are a starting guess, not statistically derived

## Roadmap

- [x] Train and evaluate a baseline churn model
- [x] Wrap it in a FastAPI service with validated request/response schemas
- [ ] Containerize with Docker
- [ ] Deploy to AWS EC2
- [ ] Track experiments and version models with MLflow

## Part of a series

This project is documented in two articles as part of my ongoing ML deployment series on Towards Data Science:

- **Part A (this repo, local build):** "My Model Worked Perfectly. Then I Tried to Make It Useful."
- **Part B (Docker + AWS deployment):** "Your Model Isn't Done Until Someone Else Can Call It"

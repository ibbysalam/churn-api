# app/main.py
from fastapi import FastAPI
from schemas import CustomerRequest, ChurnPrediction
from model import predict_churn

app = FastAPI(title="Northline Mobile Churn API")


@app.get("/")
def root():
    return {"status": "ok", "service": "churn-api"}


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerRequest):
    result = predict_churn(customer.model_dump())
    return result
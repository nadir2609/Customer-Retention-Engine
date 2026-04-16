from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CustomerFeatures, MetadataResponse, PredictionResponse, RecommendationPayload
from .service import get_metadata, get_model_artifacts, predict_customer, recommend_customer


app = FastAPI(
    title="Customer Retention Engine API",
    description="FastAPI backend for churn prediction and retention recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    artifacts = get_model_artifacts()
    return {
        "message": "Customer Retention Engine API is running.",
        "model_name": artifacts.best_model_name,
    }


@app.get("/health")
def health() -> dict[str, str]:
    artifacts = get_model_artifacts()
    return {"status": "ok", "model_name": artifacts.best_model_name}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> dict:
    return get_metadata()


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> dict:
    return predict_customer(customer)


@app.post("/recommend", response_model=RecommendationPayload)
def recommend(customer: CustomerFeatures) -> dict:
    return recommend_customer(customer)

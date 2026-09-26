# churn_model/api.py
"""
FastAPI prediction service for Customer Churn Prediction.
Endpoints:
  GET  /health         - Health check + model version info
  POST /predict        - Single customer churn prediction
  POST /predict/batch  - Batch prediction for multiple customers
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_model import config
from churn_model.predict import ChurnPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_api")

predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; release on shutdown."""
    global predictor
    try:
        predictor = ChurnPredictor(model_path=config.MODEL_PATH)
        logger.info(
            "Model artifacts loaded successfully from %s (Threshold: %.4f)",
            config.MODEL_PATH,
            predictor.threshold,
        )
    except Exception as e:
        logger.warning(
            "Model failed to load on startup: %s. Predictions will be unavailable until trained.",
            e,
        )
        predictor = None
    yield
    predictor = None
    logger.info("Model artifacts unloaded.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-grade REST API for telecom customer churn prediction. "
        "Selected by ROC AUC, tracked with MLflow, and optimized for F1 threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"] = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    Dependents: Literal["Yes", "No"] = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., examples=["No"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., examples=["DSL"])
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["Yes"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[846.0])


class PredictionResponse(BaseModel):
    predicted_churn: int = Field(..., description="1 = Will Churn, 0 = Will Stay")
    churn_probability: float = Field(..., description="Probability of churning (0–1)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")


class BatchPredictionResponse(BaseModel):
    total: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    api_version: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness and readiness health check."""
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_path=config.MODEL_PATH,
        api_version=app.version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn for a single customer.
    Returns binary prediction, calibrated probability score, and risk tier.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure models/ directory contains trained artifacts.",
        )

    try:
        record = customer.model_dump()
        result = predictor.predict_record(record)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(customers: List[CustomerInput]):
    """
    Predict churn for a batch of customers in a single request.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure models/ directory contains trained artifacts.",
        )
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Customer list must not be empty.")
    if len(customers) > 1000:
        raise HTTPException(
            status_code=400, detail="Batch size limited to 1000 customers per request."
        )

    try:
        df = pd.DataFrame([c.model_dump() for c in customers])
        result_df = predictor.predict(df)
        predictions = [
            PredictionResponse(
                predicted_churn=int(row["Predicted_Churn"]),
                churn_probability=float(row["Churn_Probability"]),
                risk_level=str(row["Risk_Tier"]),
            )
            for _, row in result_df.iterrows()
        ]
        return BatchPredictionResponse(total=len(predictions), predictions=predictions)
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Batch prediction error: {str(e)}")

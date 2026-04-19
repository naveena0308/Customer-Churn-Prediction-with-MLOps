# churn_model/api.py
"""
FastAPI prediction service for Customer Churn Prediction.
Endpoints:
  GET  /health         - Health check + model version info
  POST /predict        - Single customer churn prediction
  POST /predict/batch  - Batch prediction for multiple customers
"""

from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_model import config
from churn_model.predict import ChurnPredictor

# ── Global model holder ────────────────────────────────────────────────────────
predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; release on shutdown."""
    global predictor
    try:
        predictor = ChurnPredictor(model_path=config.MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model failed to load: {e}")
        predictor = None
    yield
    predictor = None
    print("Model unloaded.")


# ── App definition ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready REST API for telecom customer churn prediction. "
        "Powered by a RandomForest / Logistic Regression ensemble selected by ROC AUC, "
        "tracked with MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"] = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="DSL")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=846.0)

    model_config = {"json_schema_extra": {"examples": [{}]}}


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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob < 0.35:
        return "LOW"
    elif prob < 0.65:
        return "MEDIUM"
    return "HIGH"


def _customer_to_df(customer: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness + readiness check."""
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
    Returns binary prediction, probability score, and risk tier.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    df = _customer_to_df(customer)
    try:
        result = predictor.predict(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")

    prob = float(result["Churn_Probability"].iloc[0])
    pred = int(result["Predicted_Churn"].iloc[0])
    return PredictionResponse(
        predicted_churn=pred,
        churn_probability=round(prob, 4),
        risk_level=_risk_level(prob),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(customers: List[CustomerInput]):
    """
    Predict churn for a list of customers in a single call.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Customer list must not be empty.")
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000 customers per request.")

    df = pd.DataFrame([c.model_dump() for c in customers])
    try:
        result = predictor.predict(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction error: {str(e)}")

    predictions = [
        PredictionResponse(
            predicted_churn=int(row["Predicted_Churn"]),
            churn_probability=round(float(row["Churn_Probability"]), 4),
            risk_level=_risk_level(float(row["Churn_Probability"])),
        )
        for _, row in result.iterrows()
    ]
    return BatchPredictionResponse(total=len(predictions), predictions=predictions)

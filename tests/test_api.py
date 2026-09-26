import pytest
from fastapi.testclient import TestClient

from churn_model.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 846.0,
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_single_endpoint(client, valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_churn" in data
    assert data["predicted_churn"] in (0, 1)
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_batch_endpoint(client, valid_customer):
    customers = [valid_customer, valid_customer]
    response = client.post("/predict/batch", json=customers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_invalid_input_validation(client, valid_customer):
    invalid_customer = valid_customer.copy()
    invalid_customer["MonthlyCharges"] = -10.0  # Fails ge=0 validator

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422

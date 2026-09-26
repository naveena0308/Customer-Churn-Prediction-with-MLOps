# Customer Churn Prediction with MLOps

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-3.0+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

### End-to-End Machine Learning Pipeline for Customer Churn Prediction

An enterprise-grade, production-ready machine learning pipeline to predict customer churn on 7,000+ real-world telecom records. Built with principled model selection across Random Forest, Logistic Regression, and XGBoost, featuring leak-free automated preprocessing, optimal decision threshold tuning, MLflow experiment tracking, containerized FastAPI deployment, and an automated CI test suite.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture & Structure](#project-architecture--structure)
- [Model Evaluation & Results](#model-evaluation--results)
- [Installation & Quickstart](#installation--quickstart)
- [Usage Guide](#usage-guide)
  - [1. Running the Training Pipeline](#1-running-the-training-pipeline)
  - [2. Batch Prediction CLI](#2-batch-prediction-cli)
  - [3. Python API Integration](#3-python-api-integration)
  - [4. REST API Serving (FastAPI)](#4-rest-api-serving-fastapi)
  - [5. Containerized Deployment (Docker & Compose)](#5-containerized-deployment-docker--compose)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Automated Testing & CI/CD](#automated-testing--cicd)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Customer churn directly impacts recurring revenue in telecom businesses. This project provides a production-grade machine learning system to:

1. Detect churn signals early with high recall and precision.
2. Optimize the business decision threshold using the Precision-Recall curve to maximize the F1-score rather than defaulting to a naive 0.5 cutoff.
3. Eliminate train-serving skew and data leakage between training and inference.
4. Provide real-time REST API scoring and high-throughput batch prediction.

---

## Key Features

- **Leak-Free Preprocessing**: Stratified 70/15/15 split performed before preprocessing. Population statistics (medians, encoders) are learned strictly on `X_train` and persisted to guarantee zero train-serving skew for 1-sample or batch inferences.
- **Principled Model Selection**: Compares Random Forest, Logistic Regression, and XGBoost via 3-fold cross-validated hyperparameter grid searches.
- **Metric-Driven Comparison**: Evaluates on validation ROC AUC and calibrates decision thresholds via Precision-Recall curve analysis.
- **Unbiased Holdout Evaluation**: Final model is validated on a strictly held-out test split.
- **Experiment Tracking & Model Registry**: MLflow tracks all runs, hyperparameters, validation scores, out-of-sample test metrics, and model artifacts.
- **Production REST API**: FastAPI server with strict Pydantic v2 schema validation, health probes, single prediction, and batch scoring.
- **Docker & Docker Compose**: Multi-stage lightweight Docker image with non-root security and hot-swappable volume mounts for artifacts.
- **Full Pytest Suite & CI/CD**: Automated unit and API integration tests running via GitHub Actions.

---

## Project Architecture & Structure

```
Customer-Churn-Prediction-with-MLOps/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline (lint, test, docker build)
├── churn_model/
│   ├── __init__.py
│   ├── api.py                   # FastAPI REST service (/health, /predict, /predict/batch)
│   ├── config.py                # Environment configs, paths, and hyperparameters
│   ├── data_preprocessing.py   # Leak-free DataPreprocessor with state persistence
│   ├── main.py                  # End-to-end training and evaluation orchestrator
│   ├── model_training.py        # ModelTrainer with grid search and PR threshold optimization
│   ├── model_utils.py           # Robust artifact serialization and loading
│   └── predict.py               # Batch CLI and programmatic prediction interface
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Telco Churn Dataset (7,043 records)
├── models/                      # Saved production artifacts
│   ├── .gitkeep
│   ├── churn_model.pkl          # Selected best estimator (Logistic Regression)
│   ├── preprocessor.pkl         # Fitted preprocessor with learned population medians
│   ├── scaler.pkl               # Fitted StandardScaler
│   ├── label_encoders.pkl       # Fitted label encoders
│   ├── feature_columns.pkl      # Feature column schema
│   └── threshold.pkl            # Calibrated optimal decision threshold
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_model.py            # Model training & threshold optimization tests
│   └── test_preprocessing.py    # Preprocessor, data leakage, and single-sample tests
├── Dockerfile                   # Multi-stage production container build
├── docker-compose.yml           # Unified orchestration for API & MLflow server
├── EDA_Telecom_Churn.ipynb      # Exploratory Data Analysis & insight discovery
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Model Evaluation & Results

Trained on 7,043 records with class imbalance handling (`class_weight='balanced'` and `scale_pos_weight`). Decision thresholds are tuned on the validation set to maximize minority class F1.

### Validation Benchmark (GridSearchCV ROC AUC)

| Model                              | Val ROC AUC | Val Precision | Val Recall | Baseline F1 (0.50 Thresh) | Optimal F1 (PR-Tuned) | Decision Threshold |
| :--------------------------------- | :---------: | :-----------: | :--------: | :-----------------------: | :-------------------: | :----------------: |
| **Logistic Regression (Selected)** | **0.8471**  |  **0.5478**   | **0.7571** |         **0.6125**        |      **0.6357**       |     **0.5921**     |
| XGBoost                            |   0.8470    |    0.5474     |   0.8036   |           0.6080          |        0.6512         |       0.5600       |
| Random Forest                      |   0.8376    |    0.5145     |   0.8250   |           0.5940          |        0.6337         |       0.4500       |

### Unbiased Holdout Test Set Performance

Evaluated strictly on the held-out 15% test set:

- **ROC AUC**: `0.8418` (~0.845)
- **Baseline F1-Score (0.50 Threshold)**: `0.6125` (~0.61)
- **Optimal F1-Score (0.59 Threshold)**: `0.6250`
- **Test Recall**: `0.6940`
- **Test Precision**: `0.5685`
- **Test Accuracy**: `0.7786`

---

## Installation & Quickstart

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/naveena0308/Customer-Churn-Prediction-with-MLOps.git
cd Customer-Churn-Prediction-with-MLOps

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Running the Training Pipeline

Executes stratified data splitting, preprocessor fitting, model grid searches across classifiers, threshold calibration, holdout evaluation, and artifact saving:

```bash
python -m churn_model.main
```

### 2. Batch Prediction CLI

Score an entire CSV dataset directly from the terminal:

```bash
# Score a file and save predictions to CSV
python -m churn_model.predict --input data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output data/scored_customers.csv

# Run the interactive 2-sample verification demo
python -m churn_model.predict
```

### 3. Python API Integration

```python
from churn_model.predict import predict_churn, predict_churn_batch
import pandas as pd

# Single customer prediction
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
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
    "TotalCharges": 70.35
}
prediction = predict_churn(customer)
print(prediction)
# Output: {'predicted_churn': 1, 'churn_probability': 0.9105, 'risk_level': 'HIGH'}

# Batch prediction on a DataFrame
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
scored_df = predict_churn_batch(df)
```

### 4. REST API Serving (FastAPI)

Start the API server locally:

```bash
uvicorn churn_model.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at `http://localhost:8000/docs`.

#### Endpoints:

- `GET /health`: Liveness and readiness status with loaded model path.
- `POST /predict`: Real-time single customer inference with risk tier (`LOW`, `MEDIUM`, `HIGH`).
- `POST /predict/batch`: High-performance batch scoring up to 1,000 customers per request.

### 5. Containerized Deployment (Docker & Compose)

#### Using Docker Compose:

Spins up both the FastAPI prediction service and the persistent MLflow tracking server:

```bash
docker compose up --build
```

- FastAPI API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

#### Standalone Docker:

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

---

## MLflow Experiment Tracking

Launch the MLflow UI locally to inspect runs, parameters, metrics, and registered models:

```bash
mlflow ui
```

Open `http://localhost:5000` to visualize:

- Comparative ROC AUC and F1 curves across Random Forest, Logistic Regression, and XGBoost.
- Optimal threshold comparisons.
- Versioned model artifacts in the MLflow Model Registry.

---

## Automated Testing & CI/CD

Run the test suite covering data preprocessing, single-sample inference consistency, threshold optimization, and API endpoints:

```bash
pytest tests -v
```

GitHub Actions automatically runs this suite and verifies the Docker container build on every push and pull request to `main`.

---

## Configuration

Custom settings can be modified via environment variables or [`churn_model/config.py`](churn_model/config.py):

| Variable              | Default Value                               | Description                                     |
| :-------------------- | :------------------------------------------ | :---------------------------------------------- |
| `RANDOM_STATE`        | `42`                                        | Seed for reproducibility                        |
| `TEST_SIZE`           | `0.3`                                       | Test + Validation split ratio                   |
| `VAL_SIZE`            | `0.5`                                       | Split ratio between validation and test holdout |
| `DATA_PATH`           | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset path                                    |
| `MODEL_PATH`          | `models`                                    | Directory for serialized artifacts              |
| `EXPERIMENT_NAME`     | `churn_prediction`                          | MLflow experiment name                          |
| `MLFLOW_TRACKING_URI` | `None` (local `./mlruns`)                   | Remote or local tracking server URI             |

---

## License

This project is licensed under the [MIT License](LICENSE).

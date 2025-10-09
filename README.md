# Churn Prediction Project

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

### End-to-End Machine Learning Pipeline for Customer Churn Prediction

An enterprise-grade machine learning pipeline to predict customer churn for telecom companies. Built with Random Forest and Logistic Regression models, featuring comprehensive data preprocessing, MLflow experiment tracking, and model versioning.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [MLflow Integration](#mlflow-integration)
- [Configuration](#configuration)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Churn Prediction Project** is a production-ready machine learning pipeline designed to identify customers at risk of churning. The system uses advanced preprocessing techniques, multiple model architectures, and comprehensive experiment tracking to deliver accurate predictions that enable proactive customer retention strategies.

### Key Capabilities

- Automated data preprocessing with missing value handling
- Feature engineering with categorical encoding and numeric scaling
- Multiple model architectures with hyperparameter optimization
- Comprehensive model evaluation with multiple metrics
- MLflow integration for experiment tracking and reproducibility
- Modular architecture for easy deployment and maintenance
- Ready-to-use prediction interface for new customer data

---

## Features

### Data Processing
- Intelligent missing value imputation
- Categorical variable encoding (One-Hot, Label Encoding)
- Feature scaling using StandardScaler
- Automatic feature type detection

### Model Training
- Random Forest Classifier with hyperparameter tuning
- Logistic Regression with regularization
- Cross-validation for robust performance estimates
- Automatic model selection based on performance metrics

### Experiment Tracking
- MLflow integration for comprehensive experiment logging
- Model versioning and artifact management
- Hyperparameter tracking across experiments
- Performance metric visualization

### Prediction Pipeline
- Real-time churn prediction for new customers
- Probability scores for risk assessment
- Batch prediction support
- Model serving ready architecture

---

## Technology Stack

| Component           | Technology        |
|---------------------|-------------------|
| Language            | Python 3.8+       |
| ML Framework        | scikit-learn      |
| Experiment Tracking | MLflow            |
| Data Processing     | pandas, numpy     |
| Model Serialization | joblib            |
| Visualization       | matplotlib, seaborn |

---

## Project Structure

```
churn_model/
│
├── __init__.py
├── config.py              # Configuration constants and parameters
├── data_preprocessing.py  # Data preprocessing and feature engineering
├── model_training.py      # Model training and evaluation logic
├── model_utils.py         # Model persistence utilities
├── predict.py             # Prediction interface for new data
├── main.py                # Training pipeline orchestration
│
├── models/                # Saved models and preprocessors
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── data/                  # Dataset directory
│   └── telco_churn.csv
│
├── mlruns/                # MLflow experiment logs
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment tool (venv or conda)

### Setup Instructions

**1. Clone the Repository**

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

**2. Create Virtual Environment**

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Verify Installation**

```bash
python -c "import sklearn, mlflow, pandas; print('Setup successful!')"
```

---

## Usage

### Training the Model

Run the complete training pipeline:

```bash
python -m churn_model.main
```

**What this does:**
1. Loads and validates the dataset
2. Performs comprehensive data preprocessing
3. Trains Random Forest and Logistic Regression models
4. Evaluates models using multiple metrics
5. Logs experiments to MLflow
6. Saves the best performing model to `models/` directory

### Making Predictions

Predict churn for new customers:

```bash
python -m churn_model.predict
```

**Input Format:**

```python
new_customer = {
    'tenure': 12,
    'MonthlyCharges': 70.5,
    'TotalCharges': 846.0,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'PaymentMethod': 'Electronic check'
}
```

**Output:**

```python
{
    'Predicted_Churn': 1,  # 1 = Will Churn, 0 = Will Stay
    'Churn_Probability': 0.78  # 78% probability of churning
}
```

### Using the Prediction API

```python
from churn_model.predict import predict_churn

# Single prediction
result = predict_churn(customer_data)

# Batch predictions
results = predict_churn_batch(customers_dataframe)
```

---

## Model Performance

### Evaluation Metrics

The models are evaluated using comprehensive metrics:

| Metric       | Description                                  |
|--------------|----------------------------------------------|
| **Accuracy** | Overall prediction correctness               |
| **Precision**| Accuracy of positive churn predictions       |
| **Recall**   | Coverage of actual churn cases               |
| **F1-Score** | Harmonic mean of precision and recall        |
| **ROC AUC**  | Area under the receiver operating curve      |

## MLflow Integration

### Viewing Experiments

Launch the MLflow UI to explore experiments:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view:
- Experiment runs and comparisons
- Hyperparameter configurations
- Performance metrics over time
- Model artifacts and versions

### Experiment Tracking

All training runs automatically log:
- Model hyperparameters
- Performance metrics
- Feature importance scores
- Training duration
- Model artifacts

### Model Registry

Access trained models programmatically:

```python
import mlflow

# Load a specific model version
model_uri = "runs:/<run_id>/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
```

---

## Configuration

### Customizing Training Parameters

Edit `config.py` to adjust:

```python
# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5
}

LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'solver': 'lbfgs'
}

# Data processing
TEST_SIZE = 0.2
RANDOM_STATE = 42
SCALING_METHOD = 'standard'
```

### Dataset Configuration

Update the data path in `main.py`:

```python
DATA_PATH = 'data/your_dataset.csv'
TARGET_COLUMN = 'Churn'
```

---

## Future Roadmap

### Short-term Goals
- [ ] Add XGBoost and LightGBM models
- [ ] Implement feature importance visualization
- [ ] Add cross-validation with stratified K-fold
- [ ] Create automated hyperparameter tuning with Optuna

### MLOps Deployment (In Progress)
- [ ] Containerization with Docker
- [ ] CI/CD pipeline integration
- [ ] Model monitoring and drift detection
- [ ] REST API with FastAPI
- [ ] Kubernetes deployment configuration
- [ ] Automated retraining pipeline
- [ ] A/B testing framework

### Long-term Vision
- [ ] Real-time streaming predictions
- [ ] Customer segmentation integration
- [ ] Explainable AI dashboard with SHAP values
- [ ] Multi-cloud deployment support
- [ ] Advanced ensemble methods

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Support and Contact

For questions, issues, or contributions:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: naveena003office@gmail.com

---

## Acknowledgments

- Dataset: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspired by best practices from the MLOps community
- Built with open-source tools and frameworks

---

**Built with 🤖 by the Naveena Natarajan**

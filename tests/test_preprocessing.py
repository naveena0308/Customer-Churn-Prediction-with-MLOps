import numpy as np
import pandas as pd
import pytest

from churn_model.data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [0, 12, 48],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [25.0, 75.0, 95.0],
            "TotalCharges": [" ", "900.0", "4560.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_fit_and_transform(sample_raw_data):
    preprocessor = DataPreprocessor()
    transformed = preprocessor.fit_transform(sample_raw_data)

    assert "customerID" not in transformed.columns
    assert "Churn" in transformed.columns
    assert "tenure_group" in transformed.columns
    assert "charges_per_month" in transformed.columns
    assert "total_services" in transformed.columns
    assert "is_month_to_month" in transformed.columns
    assert "is_high_value" in transformed.columns

    # tenure=0 should have TotalCharges set to 0.0, not NaN
    assert not transformed["TotalCharges"].isnull().any()
    assert preprocessor.is_fitted


def test_single_sample_inference_consistency(sample_raw_data):
    preprocessor = DataPreprocessor()
    preprocessor.fit(sample_raw_data)

    single_high_val = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 50,  # Greater than median tenure (12)
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,  # Greater than median MonthlyCharges (75.0)
                "TotalCharges": 4250.0,
            }
        ]
    )

    res = preprocessor.transform(single_high_val)
    # Because medians were learned from training data, single sample evaluates is_high_value correctly!
    assert res["is_high_value"].iloc[0] == 1


def test_unseen_category_handling(sample_raw_data):
    preprocessor = DataPreprocessor()
    preprocessor.fit(sample_raw_data)

    unseen_data = pd.DataFrame(
        [
            {
                "gender": "Unknown_Gender",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "New_Option",
                "InternetService": "Satellite",
                "OnlineSecurity": "Unknown",
                "OnlineBackup": "Unknown",
                "DeviceProtection": "Unknown",
                "TechSupport": "Unknown",
                "StreamingTV": "Unknown",
                "StreamingMovies": "Unknown",
                "Contract": "Three year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Crypto",
                "MonthlyCharges": 50.0,
                "TotalCharges": 500.0,
            }
        ]
    )

    # Must not throw ValueError: y contains previously unseen labels
    res = preprocessor.transform(unseen_data)
    assert len(res) == 1

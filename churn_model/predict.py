# churn_model/predict.py
import pandas as pd
import numpy as np
from churn_model.data_preprocessing import DataPreprocessor
from churn_model.model_utils import ModelUtils
from churn_model import config


class ChurnPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        # Load trained model and all preprocessors
        (
            self.model,
            self.scaler,
            self.label_encoders,
            self.feature_columns,
            self.threshold,          # optimised F1 threshold (not always 0.5)
        ) = ModelUtils.load(model_path)

        self.preprocessor = DataPreprocessor()
        self.preprocessor.label_encoders = self.label_encoders
        self.preprocessor.scaler = self.scaler

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn on new customer data using the saved optimal threshold.

        Args:
            new_data (pd.DataFrame): Customer data in raw format (same as training CSV).

        Returns:
            pd.DataFrame: Original data + Predicted_Churn + Churn_Probability columns.
        """
        processed_data = self.preprocessor.preprocess(new_data)
        X_input = processed_data[self.feature_columns]
        X_scaled = self.scaler.transform(X_input)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Use the optimal threshold learned during training (not the naive 0.5)
        predictions = (probabilities >= self.threshold).astype(int)

        result_df = new_data.copy()
        result_df["Predicted_Churn"]    = predictions
        result_df["Churn_Probability"]  = np.round(probabilities, 4)
        return result_df


def demo():
    print("Loading model for prediction demo...")
    predictor = ChurnPredictor()
    print(f"Using decision threshold: {predictor.threshold:.4f}\n")

    sample_data = pd.DataFrame({
        "gender":           ["Female", "Male"],
        "SeniorCitizen":    [0, 1],
        "Partner":          ["Yes", "No"],
        "Dependents":       ["No", "No"],
        "tenure":           [5, 42],
        "PhoneService":     ["Yes", "Yes"],
        "MultipleLines":    ["No", "Yes"],
        "InternetService":  ["DSL", "Fiber optic"],
        "OnlineSecurity":   ["Yes", "No"],
        "OnlineBackup":     ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport":      ["No", "No"],
        "StreamingTV":      ["No", "Yes"],
        "StreamingMovies":  ["No", "Yes"],
        "Contract":         ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod":    ["Electronic check", "Mailed check"],
        "MonthlyCharges":   [70.35, 99.65],
        "TotalCharges":     [350.5, 4200.0],
    })

    results = predictor.predict(sample_data)
    print("Predictions:")
    print(results[["gender", "tenure", "Contract", "Predicted_Churn", "Churn_Probability"]])


if __name__ == "__main__":
    demo()

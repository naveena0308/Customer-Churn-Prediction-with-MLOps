# churn_model/predict.py
import argparse
import os
import sys
from typing import Dict, Union

import numpy as np
import pandas as pd

from churn_model import config
from churn_model.model_utils import ModelUtils


class ChurnPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        # Load trained model and all preprocessors
        (
            self.model,
            self.scaler,
            self.label_encoders,
            self.feature_columns,
            self.threshold,
            self.preprocessor,
        ) = ModelUtils.load(model_path)

    @staticmethod
    def _risk_tier(probability: float) -> str:
        if probability < 0.35:
            return "LOW"
        elif probability < 0.65:
            return "MEDIUM"
        return "HIGH"

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn on new customer data using learned preprocessor and optimal threshold.

        Args:
            new_data (pd.DataFrame): Customer data in raw format.

        Returns:
            pd.DataFrame: Original data + Predicted_Churn + Churn_Probability + Risk_Tier columns.
        """
        processed_data = self.preprocessor.transform(new_data)
        X_input = processed_data[self.feature_columns]
        X_scaled = self.scaler.transform(X_input)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        result_df = new_data.copy()
        result_df["Predicted_Churn"] = predictions
        result_df["Churn_Probability"] = np.round(probabilities, 4)
        result_df["Risk_Tier"] = [self._risk_tier(p) for p in probabilities]
        return result_df

    def predict_record(self, record: Dict) -> Dict:
        """Predict for a single record dictionary."""
        df = pd.DataFrame([record])
        result = self.predict(df)
        prob = float(result["Churn_Probability"].iloc[0])
        pred = int(result["Predicted_Churn"].iloc[0])
        return {
            "predicted_churn": pred,
            "churn_probability": prob,
            "risk_level": self._risk_tier(prob),
        }


# ── Global Cached Predictor & Helper Functions ──────────────────────
_cached_predictor = None


def get_predictor(model_path=config.MODEL_PATH) -> ChurnPredictor:
    global _cached_predictor
    if _cached_predictor is None:
        _cached_predictor = ChurnPredictor(model_path=model_path)
    return _cached_predictor


def predict_churn(customer_data: Union[Dict, pd.Series]) -> Dict:
    """Predict churn for a single customer dictionary or pandas Series."""
    if isinstance(customer_data, pd.Series):
        customer_data = customer_data.to_dict()
    predictor = get_predictor()
    return predictor.predict_record(customer_data)


def predict_churn_batch(customers_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Predict churn for a dataframe of customers."""
    predictor = get_predictor()
    return predictor.predict(customers_dataframe)


def demo():
    print("Loading model for prediction demo...")
    predictor = get_predictor()
    print(f"Using decision threshold: {predictor.threshold:.4f}\n")

    sample_data = pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [5, 42],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [70.35, 99.65],
            "TotalCharges": [350.5, 4200.0],
        }
    )

    results = predictor.predict(sample_data)
    print("Predictions:")
    print(
        results[
            [
                "gender",
                "tenure",
                "Contract",
                "Predicted_Churn",
                "Churn_Probability",
                "Risk_Tier",
            ]
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction CLI & Batch Inference"
    )
    parser.add_argument(
        "--input", type=str, help="Path to input CSV file for batch inference"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save output predictions CSV file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=config.MODEL_PATH,
        help="Path to saved model artifacts directory",
    )

    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist.")
            sys.exit(1)
        output_file = args.output or "predictions_output.csv"
        print(f"Running batch prediction on {args.input}...")
        df = pd.read_csv(args.input)
        predictor = ChurnPredictor(model_path=args.model_path)
        scored_df = predictor.predict(df)
        scored_df.to_csv(output_file, index=False)
        print(
            f"Successfully processed {len(df)} records. Saved predictions to {output_file}"
        )
    else:
        demo()


if __name__ == "__main__":
    main()

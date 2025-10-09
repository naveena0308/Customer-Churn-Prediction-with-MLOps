# churn_model/predict.py

import pandas as pd
import numpy as np
from churn_model.data_preprocessing import DataPreprocessor
from churn_model.model_utils import ModelUtils
from churn_model import config

class ChurnPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        # Load trained model and preprocessors
        self.model, self.scaler, self.label_encoders, self.feature_columns = ModelUtils.load(model_path)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.label_encoders = self.label_encoders
        self.preprocessor.scaler = self.scaler

    def predict(self, new_data: pd.DataFrame):
        """
        Predict churn on new customer data.
        Args:
            new_data (pd.DataFrame): Customer data in the same structure as training data.
        Returns:
            pd.DataFrame: Original data + predictions + probabilities
        """
        # Preprocess new data
        processed_data = self.preprocessor.preprocess(new_data)
        
        # Ensure consistent feature ordering
        X_input = processed_data[self.feature_columns]
        X_scaled = self.scaler.transform(X_input)

        # Predict churn
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Append predictions to data
        result_df = new_data.copy()
        result_df["Predicted_Churn"] = predictions
        result_df["Churn_Probability"] = probabilities

        return result_df

def demo():
    """
    Demo function to show usage with example input.
    Replace this with your real input CSV or API request payload.
    """
    print("Loading model for prediction demo...")
    predictor = ChurnPredictor()

    # Example new customers
    sample_data = pd.DataFrame({
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [5, 42],
        'PhoneService': ['Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No'],
        'OnlineBackup': ['No', 'Yes'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'Yes'],
        'StreamingMovies': ['No', 'Yes'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [70.35, 99.65],
        'TotalCharges': [350.5, 4200.0]
    })

    results = predictor.predict(sample_data)
    print("\nPredictions:")
    print(results[['gender', 'tenure', 'Contract', 'Predicted_Churn', 'Churn_Probability']])

if __name__ == "__main__":
    demo()

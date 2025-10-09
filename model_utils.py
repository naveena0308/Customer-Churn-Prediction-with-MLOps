# churn_model/model_utils.py
import os
import joblib

class ModelUtils:
    @staticmethod
    def save(model, scaler, label_encoders, feature_columns, path="models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(model, f"{path}/churn_model.pkl")
        joblib.dump(scaler, f"{path}/scaler.pkl")
        joblib.dump(label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump(feature_columns, f"{path}/feature_columns.pkl")
        print(f"Model and preprocessors saved to {path}/")

    @staticmethod
    def load(path="models"):
        model = joblib.load(f"{path}/churn_model.pkl")
        scaler = joblib.load(f"{path}/scaler.pkl")
        label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        feature_columns = joblib.load(f"{path}/feature_columns.pkl")
        print(f"Model and preprocessors loaded from {path}/")
        return model, scaler, label_encoders, feature_columns

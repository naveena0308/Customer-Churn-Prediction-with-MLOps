# churn_model/model_utils.py
import os
import joblib


class ModelUtils:
    @staticmethod
    def save(model, scaler, label_encoders, feature_columns,
             path="models", threshold=0.5):
        os.makedirs(path, exist_ok=True)
        joblib.dump(model,           f"{path}/churn_model.pkl")
        joblib.dump(scaler,          f"{path}/scaler.pkl")
        joblib.dump(label_encoders,  f"{path}/label_encoders.pkl")
        joblib.dump(feature_columns, f"{path}/feature_columns.pkl")
        joblib.dump(threshold,       f"{path}/threshold.pkl")   # optimal F1 threshold
        print(f"Model and preprocessors saved to {path}/  (threshold={threshold:.4f})")

    @staticmethod
    def load(path="models"):
        model           = joblib.load(f"{path}/churn_model.pkl")
        scaler          = joblib.load(f"{path}/scaler.pkl")
        label_encoders  = joblib.load(f"{path}/label_encoders.pkl")
        feature_columns = joblib.load(f"{path}/feature_columns.pkl")

        # Graceful fallback: older saved models won't have threshold.pkl
        try:
            threshold = joblib.load(f"{path}/threshold.pkl")
        except FileNotFoundError:
            threshold = 0.5

        print(f"Model loaded from {path}/  (threshold={threshold:.4f})")
        return model, scaler, label_encoders, feature_columns, threshold

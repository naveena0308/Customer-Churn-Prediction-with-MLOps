# churn_model/model_utils.py
import os

import joblib


class ModelUtils:
    @staticmethod
    def save(
        model,
        scaler,
        label_encoders,
        feature_columns,
        path="models",
        threshold=0.5,
        preprocessor=None,
    ):
        os.makedirs(path, exist_ok=True)
        joblib.dump(model, os.path.join(path, "churn_model.pkl"))
        joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(label_encoders, os.path.join(path, "label_encoders.pkl"))
        joblib.dump(feature_columns, os.path.join(path, "feature_columns.pkl"))
        joblib.dump(threshold, os.path.join(path, "threshold.pkl"))
        if preprocessor is not None:
            joblib.dump(preprocessor, os.path.join(path, "preprocessor.pkl"))
        print(f"Model and preprocessors saved to {path}/ (threshold={threshold:.4f})")

    @staticmethod
    def load(path="models"):
        model_path = os.path.join(path, "churn_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Train model first."
            )

        model = joblib.load(os.path.join(path, "churn_model.pkl"))
        scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        label_encoders = joblib.load(os.path.join(path, "label_encoders.pkl"))
        feature_columns = joblib.load(os.path.join(path, "feature_columns.pkl"))

        # Fallback for threshold
        threshold_path = os.path.join(path, "threshold.pkl")
        if os.path.exists(threshold_path):
            threshold = joblib.load(threshold_path)
        else:
            threshold = 0.5

        # Fallback for preprocessor
        prep_path = os.path.join(path, "preprocessor.pkl")
        if os.path.exists(prep_path):
            preprocessor = joblib.load(prep_path)
        else:
            from churn_model.data_preprocessing import DataPreprocessor

            preprocessor = DataPreprocessor()
            preprocessor.label_encoders = label_encoders
            preprocessor.scaler = scaler
            preprocessor.feature_columns = feature_columns
            preprocessor.is_fitted = True

        print(f"Model loaded from {path}/ (threshold={threshold:.4f})")
        return model, scaler, label_encoders, feature_columns, threshold, preprocessor

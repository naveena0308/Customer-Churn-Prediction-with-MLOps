# churn_model/data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
        self.monthly_charges_median = 0.0
        self.tenure_median = 0.0
        self.total_charges_median = 0.0

    def fit(self, df: pd.DataFrame):
        """Fit preprocessor parameters strictly on training data to avoid data leakage."""
        data = df.copy()

        if "customerID" in data.columns:
            data = data.drop("customerID", axis=1)

        # TotalCharges: tenure=0 indicates new customers with 0 billed charges
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        if "tenure" in data.columns:
            data.loc[data["tenure"] == 0, "TotalCharges"] = data.loc[
                data["tenure"] == 0, "TotalCharges"
            ].fillna(0.0)

        # Learn medians from training split
        self.total_charges_median = float(data["TotalCharges"].median())
        self.monthly_charges_median = float(data["MonthlyCharges"].median())
        self.tenure_median = float(data["tenure"].median())

        # Fit LabelEncoders for multi-class categorical columns
        categorical_cols = [
            "gender",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                le.fit(data[col].astype(str))
                self.label_encoders[col] = le

        self.is_fitted = True

        # Process training data to determine exact feature columns
        transformed = self.transform(data)
        feature_cols = [c for c in transformed.columns if c != "Churn"]
        self.feature_columns = feature_cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data using learned parameters."""
        data = df.copy()

        # ── Drop identifier column ────────────────────────────────
        if "customerID" in data.columns:
            data = data.drop("customerID", axis=1)

        # ── Fix TotalCharges ──────────────────────────────────────
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        if "tenure" in data.columns:
            data.loc[data["tenure"] == 0, "TotalCharges"] = data.loc[
                data["tenure"] == 0, "TotalCharges"
            ].fillna(0.0)
        data["TotalCharges"] = data["TotalCharges"].fillna(self.total_charges_median)

        # ── Feature Engineering ───────────────────────────────────
        # 1. Tenure group
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 72, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=True,
        ).astype(int)

        # 2. Charges per month
        data["charges_per_month"] = data["TotalCharges"] / (data["tenure"] + 1)

        # 3. Total services subscribed
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        svc_cols_present = [c for c in service_cols if c in data.columns]
        if svc_cols_present:
            svc_data = data[svc_cols_present].apply(
                lambda col: col.map(
                    lambda v: 1 if str(v).strip().lower() == "yes" else 0
                )
            )
            data["total_services"] = svc_data.sum(axis=1)
        else:
            data["total_services"] = 0

        # 4. Month-to-month flag
        if "Contract" in data.columns:
            data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype(
                int
            )

        # 5. High-value customer flag (using learned training medians, preventing skew)
        data["is_high_value"] = (
            (data["MonthlyCharges"] > self.monthly_charges_median)
            & (data["tenure"] > self.tenure_median)
        ).astype(int)

        # ── Binary encoding (Yes/No → 1/0) ────────────────────────
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        # ── Label encoding with unknown-label resilience ──────────
        categorical_cols = [
            "gender",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]
        for col in categorical_cols:
            if col in data.columns:
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    class_mapping = {
                        cls: idx for idx, cls in enumerate(encoder.classes_)
                    }
                    data[col] = (
                        data[col].astype(str).map(class_mapping).fillna(0).astype(int)
                    )
                else:
                    # In case encoder wasn't fitted yet
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le

        # ── Target encoding ───────────────────────────────────────
        if "Churn" in data.columns:
            data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

        # Ensure correct column ordering if feature_columns is established
        if self.feature_columns:
            has_churn = "Churn" in data.columns
            churn_col = data["Churn"] if has_churn else None
            # Fill any missing feature columns with 0
            for col in self.feature_columns:
                if col not in data.columns:
                    data[col] = 0
            cols = [c for c in self.feature_columns if c in data.columns]
            if has_churn:
                data = data[cols]
                data["Churn"] = churn_col
            else:
                data = data[cols]

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor on dataframe and return transformed dataframe."""
        return self.fit(df).transform(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible preprocessing hook."""
        if not self.is_fitted:
            return self.fit_transform(df)
        return self.transform(df)

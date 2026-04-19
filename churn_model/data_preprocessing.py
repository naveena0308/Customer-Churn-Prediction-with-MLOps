# churn_model/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def preprocess(self, df):
        data = df.copy()

        # ── Drop identifier column ─────────────────────────────────────────────
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)

        # ── Fix TotalCharges (stored as string with blank spaces) ──────────────
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

        # ── Feature Engineering ────────────────────────────────────────────────

        # 1. Tenure group — bucketing captures non-linear churn patterns
        #    New customers and very long-tenured customers behave differently
        data['tenure_group'] = pd.cut(
            data['tenure'],
            bins=[-1, 12, 24, 48, 72, np.inf],  # -1 ensures tenure=0 is included
            labels=[0, 1, 2, 3, 4],              # ordinal bins — safe for tree models
            right=True
        ).astype(int)

        # 2. Charges per month — proxy for service intensity / value density
        #    High TotalCharges with low tenure = expensive new customer (churn risk)
        data['charges_per_month'] = data['TotalCharges'] / (data['tenure'] + 1)

        # 3. Total services subscribed — customers with more services churn less
        service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        # Map Yes → 1, everything else (No / No internet service) → 0
        svc_data = data[service_cols].apply(
            lambda col: col.map(lambda v: 1 if str(v).strip().lower() == 'yes' else 0)
        )
        data['total_services'] = svc_data.sum(axis=1)

        # 4. Month-to-month flag — strongest single predictor of churn
        if 'Contract' in data.columns:
            data['is_month_to_month'] = (data['Contract'] == 'Month-to-month').astype(int)

        # 5. High-value customer flag — high monthly charges + long tenure
        monthly_median = data['MonthlyCharges'].median()
        tenure_median = data['tenure'].median()
        data['is_high_value'] = (
            (data['MonthlyCharges'] > monthly_median) & (data['tenure'] > tenure_median)
        ).astype(int)

        # ── Binary encoding (Yes/No → 1/0) ────────────────────────────────────
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].map({'Yes': 1, 'No': 0})

        # ── Label encoding for multi-class categorical columns ─────────────────
        categorical_cols = [
            'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[col] = self.label_encoders[col].transform(data[col].astype(str))

        # ── Target encoding ────────────────────────────────────────────────────
        if 'Churn' in data.columns:
            data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

        return data

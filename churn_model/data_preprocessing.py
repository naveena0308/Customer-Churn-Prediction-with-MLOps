# churn_model/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df):
        data = df.copy()
        
        # Handle missing TotalCharges
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
        
        # Binary encoding
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].map({'Yes': 1, 'No': 0})
        
        # Label encoding for categorical columns
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
        
        # Target encoding
        if 'Churn' in data.columns:
            data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        
        # Drop unused columns
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)
        
        return data

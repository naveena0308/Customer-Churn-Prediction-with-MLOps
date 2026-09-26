# churn_model/config.py
import os

RANDOM_STATE = 42
TEST_SIZE = 0.3
VAL_SIZE = 0.5
MODEL_PATH = os.getenv("MODEL_PATH", "models")
DATA_PATH = os.getenv("DATA_PATH", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "churn_prediction")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)

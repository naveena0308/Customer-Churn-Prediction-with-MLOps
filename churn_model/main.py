# churn_model/main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from churn_model.data_preprocessing import DataPreprocessor
from churn_model.model_training import ModelTrainer
from churn_model.model_utils import ModelUtils
from churn_model import config

def main():
    print("Loading data...")
    df = pd.read_csv(config.DATA_PATH)

    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(df)

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    preprocessor.feature_columns = X.columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE, stratify=y_temp)

    X_train_scaled = preprocessor.scaler.fit_transform(X_train)
    X_val_scaled = preprocessor.scaler.transform(X_val)
    X_test_scaled = preprocessor.scaler.transform(X_test)

    trainer = ModelTrainer()
    best_model, best_score = trainer.train(X_train_scaled, y_train, X_val_scaled, y_val, config.EXPERIMENT_NAME)

    ModelUtils.save(best_model, preprocessor.scaler, preprocessor.label_encoders, preprocessor.feature_columns, config.MODEL_PATH)

    print("Pipeline complete! Model saved.")

if __name__ == "__main__":
    main()

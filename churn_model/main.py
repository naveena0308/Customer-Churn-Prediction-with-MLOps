# churn_model/main.py
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_model import config
from churn_model.data_preprocessing import DataPreprocessor
from churn_model.model_training import ModelTrainer
from churn_model.model_utils import ModelUtils


def run_pipeline():
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"Total dataset shape: {df.shape}")

    # ── Split BEFORE preprocessing to eliminate data leakage ────────
    print("Performing stratified 70 / 15 / 15 split...")
    train_df, temp_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["Churn"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=temp_df["Churn"],
    )

    # ── Fit Preprocessor strictly on training split ─────────────────
    preprocessor = DataPreprocessor()
    train_processed = preprocessor.fit_transform(train_df)
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)

    X_train = train_processed.drop("Churn", axis=1)
    y_train = train_processed["Churn"].astype(int)

    X_val = val_processed.drop("Churn", axis=1)
    y_val = val_processed["Churn"].astype(int)

    X_test = test_processed.drop("Churn", axis=1)
    y_test = test_processed["Churn"].astype(int)

    preprocessor.feature_columns = X_train.columns.tolist()

    # ── Feature scaling fit strictly on X_train ─────────────────────
    X_train_scaled = preprocessor.scaler.fit_transform(X_train)
    X_val_scaled = preprocessor.scaler.transform(X_val)
    X_test_scaled = preprocessor.scaler.transform(X_test)

    # ── Model Training & Tuning ─────────────────────────────────────
    trainer = ModelTrainer()
    best_model, best_score, best_threshold = trainer.train(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        experiment_name=config.EXPERIMENT_NAME,
        tracking_uri=config.MLFLOW_TRACKING_URI,
        X_test=X_test_scaled,
        y_test=y_test,
    )

    # ── Unbiased Out-of-Sample Test Set Evaluation ──────────────────
    test_metrics = trainer.evaluate(
        best_model, X_test_scaled, y_test, threshold=best_threshold, prefix="test"
    )
    print("\n" + "=" * 50)
    print("[FINAL TEST SET EVALUATION - UNBIASED HOLDOUT]")
    print("=" * 50)
    print(f"Test ROC AUC   : {test_metrics['test_roc_auc']:.4f}")
    print(f"Test F1-Score  : {test_metrics['test_f1']:.4f}")
    print(f"Test Precision : {test_metrics['test_precision']:.4f}")
    print(f"Test Recall    : {test_metrics['test_recall']:.4f}")
    print(f"Test Accuracy  : {test_metrics['test_accuracy']:.4f}")
    print("=" * 50)

    # ── Save Model & Artifacts ──────────────────────────────────────
    ModelUtils.save(
        model=best_model,
        scaler=preprocessor.scaler,
        label_encoders=preprocessor.label_encoders,
        feature_columns=preprocessor.feature_columns,
        path=config.MODEL_PATH,
        threshold=best_threshold,
        preprocessor=preprocessor,
    )

    print("\nPipeline complete! Artifacts safely saved.")
    return best_model, test_metrics


if __name__ == "__main__":
    run_pipeline()

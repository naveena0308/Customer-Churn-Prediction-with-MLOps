# churn_model/model_training.py
import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_score = 0  # best ROC AUC across models
        self.best_f1 = 0  # F1 at optimal threshold for best model
        self.best_prec = 0
        self.best_rec = 0
        self.best_threshold = 0.5  # optimal decision threshold
        self.best_model_name = ""

    # ── Internal helper ───────────────────────────────────────────
    @staticmethod
    def _find_optimal_threshold(y_true, y_prob):
        """
        Scan the Precision-Recall curve to find the decision threshold
        that maximises F1-score on the validation set.

        Returns
        -------
        best_threshold : float
        best_f1        : float
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns len(thresholds) == len(precisions) - 1
        f1_scores = (
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
        )
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    def evaluate(self, model, X, y, threshold=0.5, prefix="val"):
        """Calculate metrics at a specific decision threshold."""
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            f"{prefix}_accuracy": accuracy_score(y, y_pred),
            f"{prefix}_precision": precision_score(y, y_pred, zero_division=0),
            f"{prefix}_recall": recall_score(y, y_pred, zero_division=0),
            f"{prefix}_f1": f1_score(y, y_pred, zero_division=0),
            f"{prefix}_roc_auc": roc_auc_score(y, y_prob),
            f"{prefix}_threshold": threshold,
        }
        return metrics

    # ── Main train loop ───────────────────────────────────────────
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        experiment_name="churn_prediction",
        tracking_uri=None,
        X_test=None,
        y_test=None,
    ):

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Imbalance ratio — used by XGBoost's scale_pos_weight
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        imbalance_ratio = float(neg / pos) if pos > 0 else 1.0

        # ── Model definitions ───────────────────────────────────────
        models = {
            "RandomForest": (
                RandomForestClassifier(random_state=42, class_weight="balanced"),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                },
            ),
            "LogisticRegression": (
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
                {
                    "C": [0.01, 0.1, 1.0, 10.0],
                },
            ),
            "XGBoost": (
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=imbalance_ratio,
                    verbosity=0,
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                },
            ),
        }

        # ── Training loop ───────────────────────────────────────────
        for model_name, (model, param_grid) in models.items():
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                # Hyperparameter tuning on validation ROC AUC
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_

                mlflow.log_params(grid_search.best_params_)

                # Probabilities on validation set
                y_prob_val = best_estimator.predict_proba(X_val)[:, 1]

                # Find optimal F1 decision threshold
                opt_threshold, opt_f1 = self._find_optimal_threshold(y_val, y_prob_val)
                y_pred_val = (y_prob_val >= opt_threshold).astype(int)

                val_metrics = {
                    "accuracy": accuracy_score(y_val, y_pred_val),
                    "precision": precision_score(y_val, y_pred_val, zero_division=0),
                    "recall": recall_score(y_val, y_pred_val, zero_division=0),
                    "f1_score": f1_score(y_val, y_pred_val, zero_division=0),
                    "f1_at_opt_thresh": opt_f1,
                    "roc_auc": roc_auc_score(y_val, y_prob_val),
                    "optimal_threshold": opt_threshold,
                }
                mlflow.log_metrics(val_metrics)

                # Unbiased test set evaluation if provided
                if X_test is not None and y_test is not None:
                    test_metrics = self.evaluate(
                        best_estimator,
                        X_test,
                        y_test,
                        threshold=opt_threshold,
                        prefix="test",
                    )
                    mlflow.log_metrics(test_metrics)

                try:
                    mlflow.sklearn.log_model(
                        sk_model=best_estimator,
                        name=f"{model_name.lower()}_model",
                        serialization_format="cloudpickle",
                        registered_model_name=f"churn_prediction_{model_name.lower()}",
                    )
                except Exception as log_err:
                    # Fallback for MLflow versions with different arg names
                    try:
                        mlflow.sklearn.log_model(
                            best_estimator,
                            f"{model_name.lower()}_model",
                            serialization_format="cloudpickle",
                        )
                    except Exception:
                        pass

                print(
                    f"{model_name:20s} | "
                    f"Val ROC AUC: {val_metrics['roc_auc']:.4f} | "
                    f"Val Prec: {val_metrics['precision']:.4f} | "
                    f"Val Rec: {val_metrics['recall']:.4f} | "
                    f"Val F1: {opt_f1:.4f} "
                    f"(thresh={opt_threshold:.2f})"
                )

                # Select best model based on validation ROC AUC
                if val_metrics["roc_auc"] > self.best_score:
                    self.best_score = val_metrics["roc_auc"]
                    self.best_f1 = opt_f1
                    self.best_prec = val_metrics["precision"]
                    self.best_rec = val_metrics["recall"]
                    self.best_threshold = opt_threshold
                    self.best_model = best_estimator
                    self.best_model_name = model_name

        print(
            f"\n>> Best Model : {self.best_model_name}"
            f"\n   Val ROC AUC   : {self.best_score:.4f}"
            f"\n   Val Precision : {self.best_prec:.4f}"
            f"\n   Val Recall    : {self.best_rec:.4f}"
            f"\n   Val F1 (opt)  : {self.best_f1:.4f}"
            f"\n   Threshold     : {self.best_threshold:.4f}"
        )

        return self.best_model, self.best_score, self.best_threshold

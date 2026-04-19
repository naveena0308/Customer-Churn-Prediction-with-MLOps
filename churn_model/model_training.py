# churn_model/model_training.py
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_score = 0          # best ROC AUC across models
        self.best_f1 = 0             # F1 at optimal threshold for best model
        self.best_threshold = 0.5    # optimal decision threshold
        self.best_model_name = ""

    # ── Internal helper ───────────────────────────────────────────────────────
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
            2 * precisions[:-1] * recalls[:-1]
            / (precisions[:-1] + recalls[:-1] + 1e-8)
        )
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    # ── Main train loop ───────────────────────────────────────────────────────
    def train(self, X_train, y_train, X_val, y_val,
              experiment_name="churn_prediction"):

        mlflow.set_experiment(experiment_name)

        # Imbalance ratio — used by XGBoost's scale_pos_weight
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        imbalance_ratio = neg / pos   # ≈ 2.7 for Telco dataset

        # ── Model definitions ─────────────────────────────────────────────────
        # class_weight='balanced' makes LR & RF up-weight the minority class
        # during fitting, directly addressing the 73/27 split.
        models = {
            "RandomForest": (
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced"   # FIX 1: penalise missed churners
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                }
            ),
            "LogisticRegression": (
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight="balanced",  # FIX 1: penalise missed churners
                    solver="lbfgs"
                ),
                {
                    "C": [0.01, 0.1, 1.0, 10.0],  # FIX 2: tune LR regularisation
                }
            ),
            "XGBoost": (
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    scale_pos_weight=imbalance_ratio,  # FIX 1 for XGB
                    verbosity=0,
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                }
            ),
        }

        # ── Training loop ─────────────────────────────────────────────────────
        for model_name, (model, param_grid) in models.items():
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):

                # GridSearchCV on validation ROC AUC for all models
                grid_search = GridSearchCV(
                    model, param_grid,
                    cv=3, scoring="roc_auc", n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

                mlflow.log_params(grid_search.best_params_)

                # Probabilities on validation set
                y_prob = model.predict_proba(X_val)[:, 1]

                # FIX 3: Find threshold that maximises F1 on the val set
                opt_threshold, opt_f1 = self._find_optimal_threshold(y_val, y_prob)

                # Predictions at the optimised threshold (not the default 0.5)
                y_pred = (y_prob >= opt_threshold).astype(int)

                metrics = {
                    "accuracy":          accuracy_score(y_val, y_pred),
                    "precision":         precision_score(y_val, y_pred, zero_division=0),
                    "recall":            recall_score(y_val, y_pred, zero_division=0),
                    "f1_score":          f1_score(y_val, y_pred, zero_division=0),
                    "f1_at_opt_thresh":  opt_f1,
                    "roc_auc":           roc_auc_score(y_val, y_prob),
                    "optimal_threshold": opt_threshold,
                }

                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    model,
                    f"{model_name.lower()}_model",
                    registered_model_name=f"churn_prediction_{model_name.lower()}"
                )

                print(
                    f"{model_name:20s} | "
                    f"ROC AUC: {metrics['roc_auc']:.4f} | "
                    f"F1 (opt thresh={opt_threshold:.2f}): {opt_f1:.4f}"
                )

                # Select best model by ROC AUC; tie-break on F1
                if metrics["roc_auc"] > self.best_score:
                    self.best_score = metrics["roc_auc"]
                    self.best_f1 = opt_f1
                    self.best_threshold = opt_threshold
                    self.best_model = model
                    self.best_model_name = model_name

        print(
            f"\n>> Best Model : {self.best_model_name}"
            f"\n   ROC AUC   : {self.best_score:.4f}"
            f"\n   F1 (opt)  : {self.best_f1:.4f}"
            f"\n   Threshold : {self.best_threshold:.4f}"
        )

        return self.best_model, self.best_score, self.best_threshold

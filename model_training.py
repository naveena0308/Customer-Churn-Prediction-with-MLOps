# churn_model/model_training.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""

    def train(self, X_train, y_train, X_val, y_val, experiment_name="churn_prediction"):
        mlflow.set_experiment(experiment_name)
        
        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
        }

        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                if model_name == "RandomForest":
                    param_grid = {
                        "n_estimators": [100, 200],
                        "max_depth": [10, 20, None],
                        "min_samples_split": [2, 5]
                    }
                    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    mlflow.log_params(grid_search.best_params_)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                # Metrics
                metrics = {
                    "accuracy": accuracy_score(y_val, y_pred),
                    "precision": precision_score(y_val, y_pred),
                    "recall": recall_score(y_val, y_pred),
                    "f1_score": f1_score(y_val, y_pred),
                    "roc_auc": roc_auc_score(y_val, y_pred_proba),
                }

                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    model,
                    f"{model_name.lower()}_model",
                    registered_model_name=f"churn_prediction_{model_name.lower()}"
                )

                print(f"{model_name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")

                if metrics["roc_auc"] > self.best_score:
                    self.best_score = metrics["roc_auc"]
                    self.best_model = model
                    self.best_model_name = model_name

        print(f"Best Model: {self.best_model_name} (ROC AUC: {self.best_score:.4f})")
        return self.best_model, self.best_score

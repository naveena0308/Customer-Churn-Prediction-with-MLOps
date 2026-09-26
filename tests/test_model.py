import numpy as np
import pytest

from churn_model.model_training import ModelTrainer


def test_find_optimal_threshold():
    # Synthetic ground truth and predicted probabilities
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    best_thresh, best_f1 = ModelTrainer._find_optimal_threshold(y_true, y_prob)

    assert 0.0 < best_thresh < 1.0
    assert best_f1 > 0.5


def test_model_trainer_evaluate():
    trainer = ModelTrainer()

    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

    dummy = DummyModel()
    X = np.zeros((3, 5))
    y = np.array([1, 0, 1])

    metrics = trainer.evaluate(dummy, X, y, threshold=0.5, prefix="val")

    assert "val_accuracy" in metrics
    assert "val_f1" in metrics
    assert "val_roc_auc" in metrics
    assert metrics["val_accuracy"] == 1.0

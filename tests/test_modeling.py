from pathlib import Path

import numpy as np

from src.modeling import calculate_metrics, select_threshold, train_model


def test_select_threshold_uses_f2_to_favor_recall():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.35, 0.45, 0.80, 0.30, 0.20, 0.10])
    threshold = select_threshold(y_true, probabilities)
    assert 0.30 < threshold <= 0.45


def test_calculate_metrics_returns_expected_fields():
    metrics = calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        threshold=0.5,
    )
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "average_precision",
    }
    assert metrics["accuracy"] == 1.0


def test_train_model_writes_a_complete_artifact_bundle(tmp_path):
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"

    metadata = train_model(
        data_path=Path("data/churn.csv"),
        model_dir=model_dir,
        report_dir=report_dir,
    )

    assert (model_dir / "churn_model.joblib").exists()
    assert (model_dir / "model_metadata.json").exists()
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "confusion_matrix.png").exists()
    assert (report_dir / "roc_curve.png").exists()
    assert (report_dir / "precision_recall_curve.png").exists()
    assert metadata["artifact_version"] == 1
    assert metadata["metrics"]["roc_auc"] > 0.75

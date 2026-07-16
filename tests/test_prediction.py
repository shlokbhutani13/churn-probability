from pathlib import Path

import pytest

from src.prediction import load_bundle, predict_customer


def customer_values():
    return {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }


def test_load_bundle_and_predict_customer():
    bundle = load_bundle(
        Path("models/churn_model.joblib"),
        Path("models/model_metadata.json"),
    )
    result = predict_customer(bundle, customer_values())
    assert 0 <= result.probability <= 1
    assert result.risk_band in {"Low", "Moderate", "High"}
    assert result.will_churn == (result.probability >= result.threshold)


def test_load_bundle_rejects_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact"):
        load_bundle(tmp_path / "model.joblib", tmp_path / "metadata.json")


def test_predict_customer_rejects_invalid_values():
    bundle = load_bundle(
        Path("models/churn_model.joblib"),
        Path("models/model_metadata.json"),
    )
    values = customer_values()
    values["MonthlyCharges"] = -5
    with pytest.raises(ValueError, match="MonthlyCharges"):
        predict_customer(bundle, values)

from pathlib import Path

import pandas as pd
import pytest

from src.data import FEATURES, load_dataset, validate_prediction_row


def test_load_dataset_normalizes_target_and_keeps_five_features():
    X, y = load_dataset(Path("data/churn.csv"))
    assert list(X.columns) == FEATURES
    assert set(y.unique()) == {0, 1}
    assert len(X) == 7043


def test_load_dataset_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"Churn": ["Yes", "No"], "tenure": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(path)


def test_validate_prediction_row_rejects_negative_values():
    with pytest.raises(ValueError, match="tenure"):
        validate_prediction_row(
            {
                "tenure": -1,
                "MonthlyCharges": 70.0,
                "Contract": "Month-to-month",
                "InternetService": "DSL",
                "PaymentMethod": "Electronic check",
            }
        )


def test_validate_prediction_row_rejects_unsupported_category():
    with pytest.raises(ValueError, match="Contract"):
        validate_prediction_row(
            {
                "tenure": 12,
                "MonthlyCharges": 70.0,
                "Contract": "Weekly",
                "InternetService": "DSL",
                "PaymentMethod": "Electronic check",
            }
        )

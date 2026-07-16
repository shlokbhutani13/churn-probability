from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
TARGET = "Churn"

CATEGORY_VALUES = {
    "Contract": ("Month-to-month", "One year", "Two year"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "PaymentMethod": (
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ),
}


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    required = [*FEATURES, TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    X = frame[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        if X[column].isna().any():
            raise ValueError(f"{column} contains values that cannot be converted to numbers.")

    y = (
        frame[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if y.isna().any():
        invalid = sorted(frame.loc[y.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Churn contains unsupported values: {invalid}")
    return X, y.astype(int)


def validate_prediction_row(values: Mapping[str, object]) -> pd.DataFrame:
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing prediction fields: {', '.join(missing)}")

    try:
        tenure = float(values["tenure"])
    except (TypeError, ValueError) as error:
        raise ValueError("tenure must be a number.") from error
    if tenure < 0:
        raise ValueError("tenure must be zero or greater.")

    try:
        monthly_charges = float(values["MonthlyCharges"])
    except (TypeError, ValueError) as error:
        raise ValueError("MonthlyCharges must be a number.") from error
    if monthly_charges < 0:
        raise ValueError("MonthlyCharges must be zero or greater.")

    row: dict[str, object] = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
    }
    for feature in CATEGORICAL_FEATURES:
        value = str(values[feature])
        if value not in CATEGORY_VALUES[feature]:
            allowed = ", ".join(CATEGORY_VALUES[feature])
            raise ValueError(f"{feature} must be one of: {allowed}.")
        row[feature] = value
    return pd.DataFrame([row], columns=FEATURES)

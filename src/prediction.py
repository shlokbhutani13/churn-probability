from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .data import FEATURES, validate_prediction_row
from .modeling import ARTIFACT_VERSION


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict[str, object]


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    will_churn: bool
    threshold: float
    risk_band: str


def load_bundle(model_path: Path, metadata_path: Path) -> ModelBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Model artifact version is not supported.")
    if metadata.get("feature_order") != FEATURES:
        raise ValueError("Model feature order does not match the application schema.")
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model artifact does not provide probability predictions.")
    return ModelBundle(model=model, metadata=metadata)


def _risk_band(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Moderate"
    return "High"


def predict_customer(
    bundle: ModelBundle,
    values: dict[str, object],
) -> PredictionResult:
    row = validate_prediction_row(values)
    probability = float(bundle.model.predict_proba(row)[0, 1])
    threshold = float(bundle.metadata["threshold"])
    return PredictionResult(
        probability=probability,
        will_churn=probability >= threshold,
        threshold=threshold,
        risk_band=_risk_band(probability),
    )

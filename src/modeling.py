from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    load_dataset,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACT_VERSION = 1
RANDOM_STATE = 42
C_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0)


def build_pipeline(c_value: float = 1.0) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=2, zero_division=0)
        if score > best_score or (
            np.isclose(score, best_score) and threshold > best_threshold
        ):
            best_score = float(score)
            best_threshold = float(threshold)
    return round(best_threshold, 2)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "precision": round(
            float(precision_score(y_true, predictions, zero_division=0)),
            6,
        ),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "f2": round(
            float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
            6,
        ),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "average_precision": round(
            float(average_precision_score(y_true, probabilities)),
            6,
        ),
    }


def _save_plots(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    report_dir: Path,
) -> None:
    predictions = (probabilities >= threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Stay", "Churn"],
        colorbar=False,
        cmap="Blues",
    )
    plt.title(f"Holdout confusion matrix at threshold {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y_true, probabilities, name="Logistic regression")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    plt.title("Holdout ROC curve")
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name="Logistic regression",
    )
    plt.title("Holdout precision-recall curve")
    plt.tight_layout()
    plt.savefig(report_dir / "precision_recall_curve.png", dpi=180, bbox_inches="tight")
    plt.close()


def train_model(
    *,
    data_path: Path = Path("data/churn.csv"),
    model_dir: Path = Path("models"),
    report_dir: Path = Path("reports"),
) -> dict[str, object]:
    X, y = load_dataset(data_path)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(),
        param_grid={"model__C": C_VALUES},
        scoring="roc_auc",
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    selected_model = search.best_estimator_
    training_probabilities = cross_val_predict(
        selected_model,
        X_train,
        y_train,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = select_threshold(y_train.to_numpy(), training_probabilities)
    holdout_probabilities = selected_model.predict_proba(X_holdout)[:, 1]
    metrics = calculate_metrics(y_holdout.to_numpy(), holdout_probabilities, threshold)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, model_dir / "churn_model.joblib")

    metadata: dict[str, object] = {
        "artifact_version": ARTIFACT_VERSION,
        "model_type": "class-weighted logistic regression",
        "feature_order": FEATURES,
        "threshold": threshold,
        "selected_c": float(search.best_params_["model__C"]),
        "cross_validation_roc_auc": round(float(search.best_score_), 6),
        "holdout_rows": int(len(X_holdout)),
        "training_rows": int(len(X_train)),
        "metrics": metrics,
        "random_state": RANDOM_STATE,
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_plots(
        y_holdout.to_numpy(),
        holdout_probabilities,
        threshold,
        report_dir,
    )
    return metadata

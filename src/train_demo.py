import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

BASE = Path(__file__).resolve().parents[1]  # project root

DATA_PATH = BASE / "data" / "churn.csv"  # <-- change if your file name differs
MODEL_PATH = BASE / "models" / "churn_demo_model.joblib"
COLS_PATH = BASE / "models" / "churn_demo_columns.json"
REPORT_PATH = BASE / "reports" / "demo_metrics.txt"

DEMO_COLS = ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]
TARGET_COL = "Churn"


def main():
    df = pd.read_csv(DATA_PATH)

    # Standardize target (handles Yes/No)
    y = df[TARGET_COL].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Churn column must contain Yes/No values.")

    X = df[DEMO_COLS].copy()

    numeric_cols = ["tenure", "MonthlyCharges"]
    cat_cols = ["Contract", "InternetService", "PaymentMethod"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    BASE.joinpath("models").mkdir(exist_ok=True)
    BASE.joinpath("reports").mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(COLS_PATH, "w") as f:
        json.dump(DEMO_COLS, f)

    report = classification_report(y_test, preds, digits=4)
    with open(REPORT_PATH, "w") as f:
        f.write("Demo model (5 features)\\n")
        f.write("Features: " + ", ".join(DEMO_COLS) + "\\n\\n")
        f.write(report + "\\n")
        f.write(f"ROC-AUC: {auc:.4f}\\n")

    print("Saved:", MODEL_PATH)
    print("Saved:", COLS_PATH)
    print("Saved:", REPORT_PATH)
    print("ROC-AUC:", auc)


if __name__ == "__main__":
    main()

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import load_data, normalize_target


def main():
    data_path = os.path.join("data", "churn.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Missing data/churn.csv. Put your churn dataset CSV into churn/data/churn.csv"
        )

    df = load_data(data_path)
    df = normalize_target(df)

    # Drop obvious ID columns if present
    for col in ["customerID", "CustomerID", "id", "ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df["target"].values
    X = df.drop(columns=["target"])

    # Make a copy so we can safely coerce numeric columns
    X = X.copy()

    # Try to convert columns that look numeric (e.g., "29.85") into real numbers
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.strip()

    # Coerce numeric-looking strings to numbers (non-numeric become NaN)
    X_numeric = X.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # A column is numeric if at least 80% of its values are valid numbers
    num_cols = [c for c in X.columns if X_numeric[c].notna().mean() >= 0.8]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Replace numeric columns in X with the coerced numeric versions
    X[num_cols] = X_numeric[num_cols]


    # Preprocessing pipelines
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    # Two models: baseline (LogReg) + stronger (RandomForest)
    logreg = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )),
    ])

    # Split (stratified keeps churn ratio consistent)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nClass balance:")
    print("Train:", np.bincount(y_train))
    print("Test :", np.bincount(y_test))

    # Train
    print("Training Logistic Regression (baseline)...")
    logreg.fit(X_train, y_train)

    print("Training Random Forest (stronger model)...")
    rf.fit(X_train, y_train)

    # Evaluate both
    def eval_model(name, pipe):
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        print(f"\n=== {name} ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")

        return preds, probs, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

    log_preds, log_probs, log_m = eval_model("LogReg", logreg)
    rf_preds, rf_probs, rf_m = eval_model("RandomForest", rf)

    # Pick best by ROC-AUC
    best_name, best_model = ("RandomForest", rf) if rf_m["auc"] >= log_m["auc"] else ("LogReg", logreg)
    best_probs = rf_probs if best_name == "RandomForest" else log_probs
    best_preds = rf_preds if best_name == "RandomForest" else log_preds

    print(f"\n✅ Selected best model: {best_name}")

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)
    joblib.dump(best_model, "models/churn_model.joblib")
    print("Saved: models/churn_model.joblib")

    # Reports/plots
    os.makedirs("reports", exist_ok=True)

    cm = confusion_matrix(y_test, best_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({best_name})")
    plt.savefig("reports/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y_test, best_probs)
    plt.title(f"ROC Curve ({best_name})")
    plt.savefig("reports/roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save a simple text summary
    with open("reports/metrics.txt", "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Accuracy : {accuracy_score(y_test, best_preds):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, best_preds, zero_division=0):.4f}\n")
        f.write(f"Recall   : {recall_score(y_test, best_preds, zero_division=0):.4f}\n")
        f.write(f"F1       : {f1_score(y_test, best_preds, zero_division=0):.4f}\n")
        f.write(f"ROC-AUC  : {roc_auc_score(y_test, best_probs):.4f}\n")

    print("Saved reports to reports/ (confusion_matrix.png, roc_curve.png, metrics.txt)")


if __name__ == "__main__":
    main()

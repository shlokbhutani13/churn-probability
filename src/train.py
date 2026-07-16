from __future__ import annotations

from .modeling import train_model


def main() -> None:
    metadata = train_model()
    metrics = metadata["metrics"]
    print("Saved model and evaluation artifacts.")
    print(f"Holdout ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Holdout recall: {metrics['recall']:.3f}")
    print(f"Decision threshold: {metadata['threshold']:.2f}")


if __name__ == "__main__":
    main()

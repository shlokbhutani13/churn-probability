import argparse
import joblib
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/churn_model.joblib")
    parser.add_argument("--row", required=True, help="Path to a CSV with 1 row of customer features")
    args = parser.parse_args()

    model = joblib.load(args.model)
    x = pd.read_csv(args.row)

    prob = model.predict_proba(x)[:, 1][0]
    pred = int(prob >= 0.5)

    print(f"Churn probability: {prob:.4f}")
    print(f"Prediction (1=churn, 0=stay): {pred}")

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prediction import load_bundle, predict_customer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate customer churn probability.")
    parser.add_argument("--tenure", type=float, required=True)
    parser.add_argument("--monthly-charges", type=float, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--internet-service", required=True)
    parser.add_argument("--payment-method", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_bundle(
        Path("models/churn_model.joblib"),
        Path("models/model_metadata.json"),
    )
    result = predict_customer(
        bundle,
        {
            "tenure": args.tenure,
            "MonthlyCharges": args.monthly_charges,
            "Contract": args.contract,
            "InternetService": args.internet_service,
            "PaymentMethod": args.payment_method,
        },
    )
    output = {
        "probability": round(result.probability, 6),
        "risk_band": result.risk_band,
        "threshold": result.threshold,
        "will_churn": result.will_churn,
    }
    if args.as_json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(f"Churn probability: {result.probability:.1%}")
        print(f"Risk band: {result.risk_band}")
        print(f"Flagged at threshold {result.threshold:.2f}: {result.will_churn}")


if __name__ == "__main__":
    main()

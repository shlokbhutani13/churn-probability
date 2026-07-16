from pathlib import Path

import streamlit as st

from src.data import CATEGORY_VALUES
from src.prediction import load_bundle, predict_customer

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "churn_model.joblib"
METADATA_PATH = BASE / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Churn Probability",
    page_icon="CP",
    layout="centered",
)


@st.cache_resource
def cached_bundle():
    return load_bundle(MODEL_PATH, METADATA_PATH)


st.title("Customer churn probability")
st.write(
    "Estimate retention risk from five account details. "
    "The model favors recall so a retention team misses fewer likely churners."
)

try:
    bundle = cached_bundle()
except (FileNotFoundError, ValueError) as error:
    st.error(f"The model artifacts could not be loaded. Run `python -m src.train`. {error}")
    st.stop()

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        tenure = st.number_input(
            "Tenure in months",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
            key="tenure",
        )
        contract = st.selectbox(
            "Contract",
            CATEGORY_VALUES["Contract"],
            key="contract",
        )
        payment_method = st.selectbox(
            "Payment method",
            CATEGORY_VALUES["PaymentMethod"],
            index=2,
            key="payment_method",
        )
    with right:
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=5.0,
            key="monthly_charges",
        )
        internet_service = st.selectbox(
            "Internet service",
            CATEGORY_VALUES["InternetService"],
            index=1,
            key="internet_service",
        )
    submitted = st.form_submit_button(
        "Estimate risk",
        type="primary",
        use_container_width=True,
        key="predict",
    )

if submitted:
    result = predict_customer(
        bundle,
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
        },
    )
    st.divider()
    st.metric(
        "Estimated churn probability",
        f"{result.probability:.1%}",
        border=True,
    )
    st.progress(result.probability)
    state = "error" if result.risk_band == "High" else "complete"
    with st.status(
        f"{result.risk_band} risk",
        state=state,
        expanded=True,
    ):
        decision = "is flagged" if result.will_churn else "is not flagged"
        st.write(
            f"This customer {decision} for retention outreach at the "
            f"{result.threshold:.0%} operating threshold."
        )
        st.caption(
            "This probability describes association in the training data. "
            "It does not establish why a customer may leave."
        )

st.divider()
st.subheader("Model performance")
metrics = bundle.metadata["metrics"]
metric_columns = st.columns(3)
metric_columns[0].metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
metric_columns[1].metric("Holdout recall", f"{metrics['recall']:.1%}")
metric_columns[2].metric("Holdout precision", f"{metrics['precision']:.1%}")

st.caption(
    "Educational demonstration using a public telecom churn dataset. "
    "A real retention system would require monitoring, calibration checks, and policy review."
)

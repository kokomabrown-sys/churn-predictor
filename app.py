import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

MODEL_PATH = "gradient_boosting.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
# Column order the model was trained with
FEATURES = list(model.feature_names_in_)


def build_row(tenure, monthly, total, senior, contract, internet, payment,
              paperless, tech_support, online_security) -> pd.DataFrame:
    """Turn friendly inputs into the one-hot columns the model expects."""
    row = {f: 0 for f in FEATURES}
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly
    row["TotalCharges"] = total
    row["SeniorCitizen"] = int(senior)
    for prefix, value in [
        ("Contract", contract),
        ("InternetService", internet),
        ("PaymentMethod", payment),
        ("PaperlessBilling", paperless),
        ("TechSupport", tech_support),
        ("OnlineSecurity", online_security),
    ]:
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1
    return pd.DataFrame([row])[FEATURES]


st.title("📉 Customer Churn Predictor")
st.caption("Gradient boosting model. Enter a customer's details to estimate churn risk.")

tab_one, tab_batch = st.tabs(["Single customer", "Batch (CSV)"])

with tab_one:
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)
        total = st.number_input("Total charges", 0.0, 10000.0, float(tenure * 70.0), step=10.0)
        senior = st.checkbox("Senior citizen")
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])

    if st.button("Predict", type="primary"):
        X = build_row(tenure, monthly, total, senior, contract, internet,
                      payment, paperless, tech_support, online_security)
        p = float(model.predict_proba(X)[0][1])
        st.metric("Churn probability", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= 0.5:
            st.error("High risk: this customer is likely to churn.")
        else:
            st.success("Low risk: this customer is likely to stay.")

with tab_batch:
    st.write("Upload a CSV containing these columns (already one-hot encoded):")
    st.code(", ".join(FEATURES))
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = df.copy()
            out["churn_probability"] = model.predict_proba(df[FEATURES])[:, 1]
            out["prediction"] = (out["churn_probability"] >= 0.5).map({True: "Churn", False: "Stay"})
            st.dataframe(out)
            st.download_button("Download results", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"
FALLBACK_METADATA = {
    "categories": {
        "gender": ["Female", "Male"],
        "subscription_type": ["Basic", "Premium", "Standard"],
        "contract_length": ["Monthly", "Quarterly", "Annual"],
        "age_group": ["Child", "Youth", "Adult", "Senior"],
    },
    "numeric_ranges": {
        "tenure": {"min": 1, "max": 60, "default": 32},
        "usage_frequency": {"min": 1, "max": 30, "default": 16},
        "support_calls": {"min": 0, "max": 10, "default": 3},
        "payment_delay": {"min": 0, "max": 30, "default": 12},
        "total_spend": {"min": 100, "max": 1000, "default": 661},
        "last_interaction": {"min": 1, "max": 30, "default": 14},
    },
}


def _normalize_api_url(url: str) -> str:
    return url.strip().rstrip("/")


def _load_metadata(api_url: str) -> tuple[dict[str, Any], str | None]:
    try:
        response = requests.get(f"{api_url}/metadata", timeout=5)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return FALLBACK_METADATA, f"Using local defaults because backend metadata is unavailable: {exc}"


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


st.set_page_config(page_title="Customer Retention Engine", layout="wide")
st.title("Customer Retention Engine")

with st.sidebar:
    st.header("Backend")
    api_url_input = st.text_input("FastAPI URL", value=DEFAULT_API_URL)
    api_url = _normalize_api_url(api_url_input) if api_url_input else DEFAULT_API_URL
    if st.button("Check connection"):
        try:
            health_response = requests.get(f"{api_url}/health", timeout=5)
            health_response.raise_for_status()
            payload = health_response.json()
            st.success(f"Connected. Model: {payload.get('model_name', 'unknown')}")
        except requests.RequestException as exc:
            st.error(f"Connection failed: {exc}")

metadata, metadata_error = _load_metadata(api_url)
if metadata_error:
    st.warning(metadata_error)

categories = metadata["categories"]
ranges = metadata["numeric_ranges"]

with st.form("customer_form"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", categories["gender"])
        subscription_type = st.selectbox("Subscription Type", categories["subscription_type"])
        contract_length = st.selectbox("Contract Length", categories["contract_length"])
        age_group = st.selectbox("Age Group", categories["age_group"])
        tenure = st.slider(
            "Tenure",
            min_value=int(ranges["tenure"]["min"]),
            max_value=int(ranges["tenure"]["max"]),
            value=int(ranges["tenure"]["default"]),
        )

    with right:
        usage_frequency = st.slider(
            "Usage Frequency",
            min_value=int(ranges["usage_frequency"]["min"]),
            max_value=int(ranges["usage_frequency"]["max"]),
            value=int(ranges["usage_frequency"]["default"]),
        )
        support_calls = st.slider(
            "Support Calls",
            min_value=int(ranges["support_calls"]["min"]),
            max_value=int(ranges["support_calls"]["max"]),
            value=int(ranges["support_calls"]["default"]),
        )
        payment_delay = st.slider(
            "Payment Delay",
            min_value=int(ranges["payment_delay"]["min"]),
            max_value=int(ranges["payment_delay"]["max"]),
            value=int(ranges["payment_delay"]["default"]),
        )
        total_spend = st.number_input(
            "Total Spend",
            min_value=float(ranges["total_spend"]["min"]),
            max_value=float(ranges["total_spend"]["max"]),
            value=float(ranges["total_spend"]["default"]),
            step=10.0,
        )
        last_interaction = st.slider(
            "Last Interaction",
            min_value=int(ranges["last_interaction"]["min"]),
            max_value=int(ranges["last_interaction"]["max"]),
            value=int(ranges["last_interaction"]["default"]),
        )

    submitted = st.form_submit_button("Predict and Recommend", type="primary")

if submitted:
    payload = {
        "gender": gender,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "subscription_type": subscription_type,
        "contract_length": contract_length,
        "total_spend": float(total_spend),
        "last_interaction": last_interaction,
        "age_group": age_group,
    }

    try:
        predict_response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
        predict_response.raise_for_status()
        result = predict_response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
    else:
        st.subheader("Prediction")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Class", result["prediction"])
        m2.metric("Churn Probability", _format_percent(float(result["churn_probability"])))
        m3.metric("Retention Probability", _format_percent(float(result["retention_probability"])))

        recommendation = result["recommendation"]
        st.subheader("Recommended Retention Action")
        st.write(f"**{recommendation['status']}** - {recommendation['message']}")

        if recommendation.get("original_churn_prob") is not None and recommendation.get("new_churn_prob") is not None:
            r1, r2 = st.columns(2)
            r1.metric("Churn Probability (Before)", _format_percent(float(recommendation["original_churn_prob"])))
            r2.metric("Churn Probability (After)", _format_percent(float(recommendation["new_churn_prob"])))
        elif recommendation.get("churn_probability") is not None:
            st.metric("Current Churn Probability", _format_percent(float(recommendation["churn_probability"])))

        changes = recommendation.get("recommendations") or {}
        if changes:
            rows = []
            for feature, change in changes.items():
                rows.append(
                    {
                        "Feature": feature,
                        "From": change["from"],
                        "To": change["to"],
                        "Delta": change["delta"],
                        "Direction": change["direction"],
                        "Unit": change["unit"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

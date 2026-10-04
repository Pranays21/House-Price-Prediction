"""Streamlit demo: predict a house price from its features.

Run from the project root::

    streamlit run app.py

Requires models/best_model.pkl (created by `python -m src.train`).
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.predict import load_artifacts

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Predictor")
st.write("Enter the property features and get an estimated sale price from the "
         "best trained regression model.")

YES_NO = ["yes", "no"]
FURNISHING = ["furnished", "semi-furnished", "unfurnished"]


@st.cache_resource
def get_pipeline():
    if not Path(config.BEST_MODEL_FILE).exists():
        return None
    return load_artifacts()[0]


pipeline = get_pipeline()
if pipeline is None:
    st.error("Model not found. Train it first with `python -m src.train` from the project root.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=50)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=0)
    stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
    parking = st.selectbox("Parking spaces", [0, 1, 2, 3], index=1)
    furnishingstatus = st.selectbox("Furnishing", FURNISHING, index=1)
with col2:
    mainroad = st.selectbox("Main road access", YES_NO, index=0)
    guestroom = st.selectbox("Guest room", YES_NO, index=1)
    basement = st.selectbox("Basement", YES_NO, index=1)
    hotwaterheating = st.selectbox("Hot water heating", YES_NO, index=1)
    airconditioning = st.selectbox("Air conditioning", YES_NO, index=1)
    prefarea = st.selectbox("Preferred area", YES_NO, index=1)

if st.button("Predict price", type="primary"):
    from src.features import add_engineered_features

    row = pd.DataFrame([{
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "stories": stories, "mainroad": mainroad, "guestroom": guestroom,
        "basement": basement, "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning, "parking": parking,
        "prefarea": prefarea, "furnishingstatus": furnishingstatus,
    }])
    row = add_engineered_features(row)
    price = float(pipeline.predict(row)[0])
    st.success(f"Estimated price: **₹ {price:,.0f}**")

st.divider()
st.subheader("Model leaderboard")
lb_path = Path(config.LEADERBOARD_FILE)
if lb_path.exists():
    lb = pd.read_csv(lb_path)[["model", "rmse", "mae", "r2"]]
    st.dataframe(lb.style.format({"rmse": "{:,.0f}", "mae": "{:,.0f}", "r2": "{:.4f}"}),
                 use_container_width=True)
else:
    st.info("Leaderboard will appear here after training (`python -m src.train`).")

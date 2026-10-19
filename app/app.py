"""Streamlit app. Run from the project root:  streamlit run app/app.py"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Risk Predictor")
st.caption("Educational project. Not a medical device and not medical advice.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 90, 54)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female")
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3],
                      format_func=lambda v: ["Typical angina", "Atypical angina",
                                             "Non-anginal pain", "Asymptomatic"][v])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 245)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                       format_func=lambda v: "Yes" if v else "No")
    restecg = st.selectbox("Resting ECG result", [0, 1, 2],
                           format_func=lambda v: ["Normal", "ST-T abnormality",
                                                  "LV hypertrophy"][v])
with col2:
    thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-induced angina", [0, 1],
                         format_func=lambda v: "Yes" if v else "No")
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2],
                         format_func=lambda v: ["Upsloping", "Flat", "Downsloping"][v])
    ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                        format_func=lambda v: ["Null", "Fixed defect", "Normal",
                                               "Reversible defect"][v])

if st.button("Predict risk", type="primary"):
    row = pd.DataFrame([dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                             restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                             slope=slope, ca=ca, thal=thal)])
    p = float(pipe.predict_proba(row)[0, 1])
    st.metric("Estimated probability of heart disease", f"{p:.0%}")
    st.progress(p)
    if p >= 0.5:
        st.warning("Higher risk predicted. Please consult a doctor for a proper evaluation.")
    else:
        st.success("Lower risk predicted. This does not replace a medical check-up.")
    st.caption(f"Model used: {bundle['name']}")

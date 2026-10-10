"""
Streamlit Web Application: Student Performance Prediction
Direct entrypoint matching Step 5 specification.
Loads model.pkl and provides interactive sliders, Predict button, and results.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

st.title("🎓 Student Performance Prediction App")
st.write("Predict whether a student will **Pass** or be **At Academic Risk** using Machine Learning.")

# Load model
@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_trained_model()

if pipeline is None:
    st.error("model.pkl not found! Please run 'python main.py' first.")
else:
    st.subheader("Enter Student Academic Information:")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Weekly Study Hours", min_value=1.0, max_value=40.0, value=18.0, step=0.5)
        attendance = st.slider("Attendance Rate (%)", min_value=40.0, max_value=100.0, value=85.0, step=1.0)
        previous_score = st.slider("Previous Exam Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        assignment_rate = st.slider("Assignment Completion Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

    with col2:
        sleep_hours = st.slider("Nightly Sleep Hours", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
        extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], index=0)
        tutoring = st.radio("Tutoring Classes", ["No", "Yes"], index=0)
        parental_edu = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)

    input_data = pd.DataFrame([{
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance,
        "previous_exam_score": previous_score,
        "assignment_completion_rate": assignment_rate,
        "sleep_hours": sleep_hours,
        "extracurricular_activities": extracurricular,
        "tutoring_classes": tutoring,
        "parental_education_level": parental_edu
    }])

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        pred = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0][1]

        st.divider()
        st.subheader("Prediction Result:")

        if pred == 1:
            st.success(f"✅ Prediction: **PASS / ON TRACK** (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"⚠️ Prediction: **FAIL / AT ACADEMIC RISK** (Confidence: {(1-prob)*100:.1f}%)")

        st.progress(prob, text=f"Passing Probability: {prob*100:.1f}%")

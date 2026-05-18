# =========================================================
# app.py
# CarePredict AI
# Full AI Diabetes + Sepsis Prediction Dashboard
# Streamlit Single File Application
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CarePredict AI",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f6f9;
}

h1, h2, h3 {
    color: #003049;
}

.stButton>button {
    background-color: #0077b6;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stDownloadButton>button {
    background-color: #2a9d8f;
    color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================

diabetes_model = joblib.load("diabetes_model.pkl")
sepsis_model = joblib.load("sepsis_model.pkl")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 CarePredict AI")

prediction_type = st.sidebar.selectbox(
    "Select Prediction Type",
    [
        "Diabetes Prediction",
        "Sepsis Prediction"
    ]
)

# =========================================================
# PDF REPORT FUNCTION
# =========================================================

def generate_pdf_report(
    prediction_name,
    probability,
    status,
    patient_data
):

    os.makedirs("reports", exist_ok=True)

    pdf_path = "reports/patient_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    # =====================================================
    # TITLE
    # =====================================================

    title = Paragraph(
        "AI Patient Risk Prediction Report",
        styles["Title"]
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = Paragraph(
        f"""
        <b>Prediction Type:</b> {prediction_name}<br/>
        <b>Risk Probability:</b> {probability:.2f}%<br/>
        <b>Status:</b> {status}<br/>
        """,
        styles["BodyText"]
    )

    elements.append(summary)

    elements.append(Spacer(1, 20))

    # =====================================================
    # PATIENT DATA TABLE
    # =====================================================

    table_data = [["Feature", "Value"]]

    for key, value in patient_data.items():

        table_data.append([
            key,
            str(value)
        ])

    table = Table(table_data)

    table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("GRID", (0, 0), (-1, -1), 1, colors.black),

        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

        ("ALIGN", (0, 0), (-1, -1), "CENTER"),

    ]))

    elements.append(table)

    doc.build(elements)

    return pdf_path

# =========================================================
# DIABETES PREDICTION
# =========================================================

if prediction_type == "Diabetes Prediction":

    st.title("🩸 AI Diabetes Risk Prediction Dashboard")

    st.markdown(
        "Predict diabetes risk using explainable AI"
    )

    col1, col2 = st.columns(2)

    # =====================================================
    # INPUTS
    # =====================================================

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0
        )

        age = st.number_input(
            "Age",
            min_value=1
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("Predict Diabetes Risk"):

        input_data = pd.DataFrame({

            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree],
            "Age": [age]

        })

        prediction = diabetes_model.predict(
            input_data
        )[0]

        probability = diabetes_model.predict_proba(
            input_data
        )[0][1]

        # =================================================
        # RESULT
        # =================================================

        st.metric(
            "Diabetes Risk Probability",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:

            status = "HIGH RISK"

            st.error("⚠ HIGH RISK OF DIABETES")

        else:

            status = "LOW RISK"

            st.success("✅ LOW RISK OF DIABETES")

        # =================================================
        # AI INTERPRETATION
        # =================================================

        st.info(f"""
AI Clinical Interpretation

• Risk Level: {status}
• Probability: {probability*100:.2f}%

Recommendations:
- Monitor glucose regularly
- Lifestyle modifications
- HbA1c testing
- Endocrinology consultation
""")

        # =================================================
        # SHAP EXPLAINABILITY
        # =================================================

        st.subheader("🔍 SHAP Explainability")

        explainer = shap.TreeExplainer(
            diabetes_model
        )

        shap_values = explainer.shap_values(
            input_data
        )

        # =================================================
        # SHAP BAR PLOT
        # =================================================

        st.subheader("📊 Feature Importance")

        fig, ax = plt.subplots()

        shap.summary_plot(
            shap_values,
            input_data,
            plot_type="bar",
            show=False
        )

        st.pyplot(fig)

        # =================================================
        # SHAP WATERFALL
        # =================================================

        st.subheader("🧠 Patient-Level Explanation")

        fig2 = plt.figure()

        shap.plots._waterfall.waterfall_legacy(

            explainer.expected_value,

            shap_values[0],

            feature_names=input_data.columns

        )

        st.pyplot(fig2)

        # =================================================
        # RISK PIE CHART
        # =================================================

        risk_df = pd.DataFrame({
            "Category": [
                "Low Risk",
                "High Risk"
            ],
            "Value": [
                100 - probability*100,
                probability*100
            ]
        })

        fig3 = px.pie(
            risk_df,
            names="Category",
            values="Value",
            title="Diabetes Risk Distribution"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # =================================================
        # PDF REPORT
        # =================================================

        patient_data = {

            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "Blood Pressure": blood_pressure,
            "Skin Thickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "Diabetes Pedigree": diabetes_pedigree,
            "Age": age

        }

        pdf_path = generate_pdf_report(
            "Diabetes Prediction",
            probability*100,
            status,
            patient_data
        )

        with open(pdf_path, "rb") as file:

            st.download_button(
                label="⬇ Download PDF Report",
                data=file,
                file_name="diabetes_report.pdf",
                mime="application/pdf"
            )

# =========================================================
# SEPSIS PREDICTION
# =========================================================

elif prediction_type == "Sepsis Prediction":

    st.title("🚨 AI Sepsis Prediction Dashboard")

    st.markdown(
        "Predict sepsis risk using explainable AI"
    )

    col1, col2 = st.columns(2)

    # =====================================================
    # INPUTS
    # =====================================================

    with col1:

        heart_rate = st.number_input(
            "Heart Rate",
            min_value=0
        )

        temperature = st.number_input(
            "Temperature",
            min_value=0.0
        )

        respiratory_rate = st.number_input(
            "Respiratory Rate",
            min_value=0
        )

    with col2:

        oxygen_saturation = st.number_input(
            "Oxygen Saturation",
            min_value=0
        )

        wbc = st.number_input(
            "WBC Count",
            min_value=0
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("Predict Sepsis Risk"):

        # Build input row that matches the trained model's feature names.
        # Obtain expected feature names from the loaded model (XGBoost stores them on the booster).
        try:
            expected_features = sepsis_model.get_booster().feature_names
        except Exception:
            expected_features = list(getattr(sepsis_model, "feature_names_in_", []) or [])

        # If we were able to read expected features, create a full row filled with zeros
        # and then map the few UI inputs to their likely training feature names.
        if expected_features:
            row = {f: 0 for f in expected_features}

            # Common mappings from the UI input labels to training feature names
            mappings = {
                "hr_mean": heart_rate,
                "temp_celsius_mean": temperature,
                "respiratory_rate_mean": respiratory_rate,
                "spo2_mean": oxygen_saturation,
                "wbc": wbc
            }

            # Populate the row when the target name exists in expected features.
            for tgt_name, val in mappings.items():
                if tgt_name in row:
                    row[tgt_name] = val
                # also support simple capitalized variants if present
                elif tgt_name.capitalize() in row:
                    row[tgt_name.capitalize()] = val

            input_data = pd.DataFrame([row], columns=expected_features)
        else:
            # Fallback: if we couldn't determine expected features, use the small UI schema.
            input_data = pd.DataFrame({
                "HeartRate": [heart_rate],
                "Temperature": [temperature],
                "RespiratoryRate": [respiratory_rate],
                "OxygenSaturation": [oxygen_saturation],
                "WBC": [wbc]
            })

        prediction = sepsis_model.predict(
            input_data
        )[0]

        probability = sepsis_model.predict_proba(
            input_data
        )[0][1]

        # =================================================
        # RESULT
        # =================================================

        st.metric(
            "Sepsis Risk Probability",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:

            status = "CRITICAL"

            st.error("🚨 CRITICAL SEPSIS RISK")

        else:

            status = "LOW RISK"

            st.success("✅ LOW SEPSIS RISK")

        # =================================================
        # AI INTERPRETATION
        # =================================================

        st.info(f"""
AI Clinical Interpretation

• Risk Level: {status}
• Probability: {probability*100:.2f}%

Recommendations:
- Immediate physician review
- Monitor vitals closely
- Evaluate infection markers
- Consider ICU assessment
""")

        # =================================================
        # SHAP EXPLAINABILITY
        # =================================================

        st.subheader("🔍 SHAP Explainability")

        explainer = shap.TreeExplainer(
            sepsis_model
        )

        shap_values = explainer.shap_values(
            input_data
        )

        # =================================================
        # SHAP BAR PLOT
        # =================================================

        st.subheader("📊 Feature Importance")

        fig, ax = plt.subplots()

        shap.summary_plot(
            shap_values,
            input_data,
            plot_type="bar",
            show=False
        )

        st.pyplot(fig)

        # =================================================
        # SHAP WATERFALL
        # =================================================

        st.subheader("🧠 Patient-Level Explanation")

        fig2 = plt.figure()

        shap.plots._waterfall.waterfall_legacy(

            explainer.expected_value,

            shap_values[0],

            feature_names=input_data.columns

        )

        st.pyplot(fig2)

        # =================================================
        # RISK PIE CHART
        # =================================================

        risk_df = pd.DataFrame({
            "Category": [
                "Low Risk",
                "High Risk"
            ],
            "Value": [
                100 - probability*100,
                probability*100
            ]
        })

        fig3 = px.pie(
            risk_df,
            names="Category",
            values="Value",
            title="Sepsis Risk Distribution"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # =================================================
        # PDF REPORT
        # =================================================

        patient_data = {

            "Heart Rate": heart_rate,
            "Temperature": temperature,
            "Respiratory Rate": respiratory_rate,
            "Oxygen Saturation": oxygen_saturation,
            "WBC": wbc

        }

        pdf_path = generate_pdf_report(
            "Sepsis Prediction",
            probability*100,
            status,
            patient_data
        )

        with open(pdf_path, "rb") as file:

            st.download_button(
                label="⬇ Download PDF Report",
                data=file,
                file_name="sepsis_report.pdf",
                mime="application/pdf"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<center>🩺 CarePredict AI | Explainable Healthcare AI Dashboard</center>",
    unsafe_allow_html=True
)
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import random

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

def generate_pdf(patient_name, diagnosis, probability, data_dict, note_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "❤️ Heart Disease Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Patient Name: {patient_name}")
    c.drawString(50, height - 100, f"Diagnosis: {diagnosis}")
    c.drawString(50, height - 120, f"Model Confidence: {probability:.2%}")

    y = height - 160
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Input Summary:")
    y -= 20

    c.setFont("Helvetica", 11)
    for key, value in data_dict.items():
        c.drawString(60, y, f"- {key}: {value}")
        y -= 18
        if y < 80:
            c.showPage()
            y = height - 50

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Doctor's Note:")
    y -= 20
    c.setFont("Helvetica", 11)
    for line in note_text.splitlines():
        c.drawString(60, y, line)
        y -= 18
        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# Streamlit page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.markdown("Provide the patient's details to assess heart disease risk. No abbreviations used in this app.")

# Input form
with st.form("heart_form"):
    st.subheader("🧍 Patient Information")
    patient_name = st.text_input("Full Name", placeholder="Enter patient full name")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (in years)", 28, 77, 53)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 100, 603, 200)
        fasting_bs = st.radio("Fasting Blood Sugar over 120 mg/dL?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 202, 140)
        oldpeak = st.number_input("ST Depression from Exercise", -2.6, 6.2, 1.0, step=0.1)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Atypical Angina", "Asymptomatic", "Non-Anginal Pain", "Typical Angina"])
        resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
        exercise_angina = st.radio("Exercise-Induced Chest Pain", ["No", "Yes"])
        st_slope = st.selectbox("Slope of Peak ST Segment", ["Upsloping", "Flat", "Downsloping"])

    submit = st.form_submit_button("🔍 Predict")

# Predict
if submit:
    if not patient_name.strip():
        st.warning("⚠️ Please enter the patient's full name to generate the report.")
    else:
        # Encoding
        mapping = {
            'Sex': {'Male': 1, 'Female': 0},
            'ChestPainType': {
                'Atypical Angina': 0,
                'Asymptomatic': 1,
                'Non-Anginal Pain': 2,
                'Typical Angina': 3
            },
            'RestingECG': {
                'Left Ventricular Hypertrophy': 0,
                'Normal': 1,
                'ST-T wave abnormality': 2
            },
            'ExerciseAngina': {'No': 0, 'Yes': 1},
            'ST_Slope': {
                'Downsloping': 0,
                'Flat': 1,
                'Upsloping': 2
            }
        }

        input_data = np.array([[
            age,
            mapping['Sex'][sex],
            mapping['ChestPainType'][chest_pain],
            resting_bp,
            cholesterol,
            fasting_bs,
            mapping['RestingECG'][resting_ecg],
            max_hr,
            mapping['ExerciseAngina'][exercise_angina],
            oldpeak,
            mapping['ST_Slope'][st_slope]
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Risk Detected: Heart disease likely.\n\n**Confidence: {probability:.2%}**")
            diagnosis = "Positive (Heart Disease Detected)"
        else:
            st.success(f"✅ No heart disease detected.\n\n**Confidence: {probability:.2%}**")
            diagnosis = "Negative (No Heart Disease)"

        # Prepare data dictionary
        patient_data = {
            "Age": f"{age} years",
            "Sex": sex,
            "Chest Pain Type": chest_pain,
            "Resting Blood Pressure": f"{resting_bp} mm Hg",
            "Serum Cholesterol": f"{cholesterol} mg/dL",
            "Fasting Blood Sugar > 120 mg/dL": "Yes" if fasting_bs == 1 else "No",
            "Resting ECG": resting_ecg,
            "Maximum Heart Rate": max_hr,
            "Exercise-Induced Chest Pain": exercise_angina,
            "ST Depression": oldpeak,
            "ST Segment Slope": st_slope
        }

        # 📈 Graph: Bar chart of numeric values
        chart_data = pd.DataFrame({
            "Feature": list(patient_data.keys()),
            "Value": list(patient_data.values())
        })
        numeric_chart_data = chart_data[chart_data['Value'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        numeric_chart_data["Value"] = numeric_chart_data["Value"].astype(float)

        st.markdown("### 📈 Patient Feature Overview")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(numeric_chart_data["Feature"], numeric_chart_data["Value"], color="#3498db")
        ax.invert_yaxis()
        ax.set_xlabel("Value")
        ax.set_title("Patient Input Summary")
        st.pyplot(fig)

        # Notes and PDF
        st.markdown("---")
        st.subheader("📝 Doctor's Note")
        note_text = st.text_area("Write a note to include in the PDF report (optional)")

        # Generate PDF
        safe_name = patient_name.strip().replace(' ', '_') or "patient"
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_num = random.randint(1000, 9999)
        filename = f"{safe_name}_heart_report_{now_str}_{rand_num}.pdf"

        st.write(f"✅ Report filename will be: **{filename}**")

        pdf_file = generate_pdf(patient_name, diagnosis, probability, patient_data, note_text)

        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_file,
            file_name=filename,
            mime="application/pdf"
        )

# Abbreviations section
with st.expander("📘 View Abbreviations and Full Forms"):
    st.markdown("""
    | Term                          | Full Meaning                                  |
    |------------------------------|-----------------------------------------------|
    | ST Depression (Oldpeak)      | ST segment depression during exercise         |
    | ECG                          | Electrocardiogram                             |
    | Chest Pain Type              | Type of chest pain (Atypical, Typical, etc.)  |
    | Max Heart Rate               | Maximum heart rate during stress test         |
    | Slope                        | Shape of ST segment during exercise           |
    | Fasting Blood Sugar >120     | Indicates diabetes risk if 'Yes'              |
    | Angina                       | Chest pain due to restricted blood flow       |
    """)
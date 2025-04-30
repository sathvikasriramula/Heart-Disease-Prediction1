import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown(
    "<h1 style='color: white;'>💓 Enter Patient Details</h1>", unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 29, 77, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"
    ])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Cholesterol (mg/dL)", 100, 600, 245)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])

with col2:
    restecg = st.selectbox("Resting ECG", [
        "Normal (0)", "ST-T Abnormality (1)", "LVH (2)"
    ])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.radio("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment", [
        "Upsloping (0)", "Flat (1)", "Downsloping (2)"
    ])
    ca = st.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [
        "Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"
    ])

if st.button("🔎 Predict Risk"):
    # Map inputs
    sex = 1 if sex == "Male" else 0
    cp = int(cp.split("(")[-1].replace(")", ""))
    fbs = 1 if fbs == "Yes" else 0
    restecg = int(restecg.split("(")[-1].replace(")", ""))
    exang = 1 if exang == "Yes" else 0
    slope = int(slope.split("(")[-1].replace(")", ""))
    thal = int(thal.split("(")[-1].replace(")", ""))

    # Make prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease")


import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Diabetes Predictor"])

# --- Home Page ---
if page == "Home":
    with open("diabetes_landing_page.html", "r") as f:
        html = f.read()
    components.html(html, height=800, scrolling=True)

# --- Predictor Page ---
elif page == "Diabetes Predictor":
    st.title("Diabetes Risk Predictor")
    st.write("Enter patient info to predict diabetes risk:")

    # Load model
    model = joblib.load("diabetes_model.pkl")

    # Input form
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucose Level", 0, 200, 120)
    blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 10, 100, 33)

    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]],
                            columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        result = "High Risk of Diabetes" if prediction == 1 else "Low Risk"
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {probability * 100:.2f}%")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    # Handle missing values
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

# Load data and train model
df = load_data()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# App title
st.title("Diabetes Risk Predictor")
st.write("Enter patient details to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 0, 200, 120)
bp = st.slider("Blood Pressure", 0, 140, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 33)

# Predict
input_df = pd.DataFrame([[
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
]], columns=X.columns)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    result = "High Risk of Diabetes" if prediction == 1 else "Low Risk"
    st.subheader(f"Prediction: {result}")
    #st.write(f"Confidence: {probability * 100:.2f}%")

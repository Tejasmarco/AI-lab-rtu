import streamlit as st
import pandas as pd
import joblib
model = joblib.load("heart_model.pkl")
st.title("Heart Disease Prediction")
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
sex_num = 1 if sex == "Male" else 0
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 0)
input_data = pd.DataFrame([[age, sex_num, cp, 120, 200, 0, 1, 150, 0, 1.0, 1, 0, 2]],
                          columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.error(f"High risk! Confidence: {prob:.2%}")
    else:
        st.success(f"Low risk! Confidence: {prob:.2%}")

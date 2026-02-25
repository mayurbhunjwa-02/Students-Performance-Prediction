import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("model.pkl","rb"))

st.title("🎓 Student Performance Prediction")

st.write("Enter student details")

# inputs
study_hour = st.number_input("Study Hour", 0.0, 18.0)
attendence = st.number_input("Attendence (%)", 0.0, 100.0)
mathscore = st.number_input("Math Score", 0.0, 100.0)

if st.button("Predict"):

    features = np.array([[study_hour, attendence, mathscore]])

    prediction = model.predict(features)[0]

    st.subheader(f"Predicted Final Grade: {round(prediction,2)}")

    # performance category
    if prediction >= 70:
        st.success("Excellent Performance")
    elif prediction >= 60:
        st.warning("Average Performance")
    else:
        st.error("Low Performance")


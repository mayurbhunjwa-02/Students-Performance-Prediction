import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Prediction")
st.write("Enter student details to predict Final Grade")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    study_hour = st.number_input(
        "Study Hour",
        min_value=0.0,
        placeholder="Enter study hours"
    )

    attendence = st.number_input(
        "Attendence (%)",
        min_value=0.0,
        max_value=100.0,
        placeholder="Enter attendence"
    )

with col2:
    mathscore = st.number_input(
        "Math Score",
        min_value=0.0,
        max_value=100.0,
        placeholder="Enter math score"
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Predict Final Grade 🚀"):

    input_data = np.array([[study_hour, attendence, mathscore]])

    prediction = model.predict(input_data)

    result = prediction[0]

    st.success(f"📊 Predicted Final Grade: {round(result,2)}")

    # Extra feedback (looks professional in presentation)
    if result >= 75:
        st.info("🌟 Excellent Performance Expected")
    elif result >= 50:
        st.warning("👍 Average Performance Expected")
    else:
        st.error("⚠️ Performance May Be Low")
import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Prediction")
st.write("Enter student details to predict Final Grade")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    study_hour = st.number_input(
        "Study Hour",
        min_value=0.0,
        placeholder="Enter study hours"
    )

    attendence = st.number_input(
        "Attendence (%)",
        min_value=0.0,
        max_value=100.0,
        placeholder="Enter attendence"
    )

with col2:
    mathscore = st.number_input(
        "Math Score",
        min_value=0.0,
        max_value=100.0,
        placeholder="Enter math score"
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Predict Final Grade 🚀"):

    input_data = np.array([[study_hour, attendence, mathscore]])

    prediction = model.predict(input_data)

    result = prediction[0]

    st.success(f"📊 Predicted Final Grade: {round(result,2)}")

    # Extra feedback (looks professional in presentation)
    if result >= 75:
        st.info("🌟 Excellent Performance Expected")
    elif result >= 50:
        st.warning("👍 Average Performance Expected")
    else:
        st.error("⚠️ Performance May Be Low")
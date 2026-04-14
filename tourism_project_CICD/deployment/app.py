import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# =========================
# Load Model from Hugging Face
# =========================
model_path = hf_hub_download(
    repo_id="Satyanjay/tourism-package-prediction-CICD-model",
    filename="tourism_model.joblib"
)
model = joblib.load(model_path)

# =========================
# Streamlit UI
# =========================
st.title("🌍 Wellness Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the
**Wellness Tourism Package** based on their profile and interaction data.
""")

# =========================
# User Inputs
# =========================

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

Age = st.number_input("Age", min_value=18, max_value=70, value=30)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)

NumberOfTrips = st.number_input("Trips per Year", min_value=0, max_value=20, value=2)
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
NumberOfChildrenVisiting = st.number_input("Children Visiting", min_value=0, max_value=5, value=0)

PreferredPropertyStar = st.selectbox("Preferred Hotel Rating", [3, 4, 5])
Passport = st.selectbox("Passport", [0, 1])
OwnCar = st.selectbox("Own Car", [0, 1])

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
NumberOfFollowups = st.slider("Number of Followups", 0, 10, 2)
DurationOfPitch = st.slider("Pitch Duration (minutes)", 5, 60, 20)

# =========================
# Convert Input to DataFrame
# =========================

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# =========================
# Prediction
# =========================

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    result = "Customer WILL Purchase" if prediction == 1 else "Customer will NOT Purchase"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

    st.write("### Input Data")
    st.dataframe(input_data)

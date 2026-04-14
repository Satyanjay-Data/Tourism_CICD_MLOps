import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# =========================
# Load Model from Hugging Face
# =========================
model_path = hf_hub_download(
    repo_id="Satyanjay/tourism-package-prediction-model",
    filename="tourism_model.joblib"   
)

model = joblib.load(model_path)

# =========================
# App UI
# =========================
st.title("Wellness Tourism Package Prediction App")

st.write("""
Predict whether a customer will purchase the **Wellness Tourism Package**
based on their profile and interaction behavior.
""")

# =========================
# Inputs
# =========================
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])

Age = st.number_input("Age", 18, 70, 30)
MonthlyIncome = st.number_input("Monthly Income", 0, 200000, 50000)

NumberOfTrips = st.number_input("Trips per Year", 0, 20, 2)
NumberOfPersonVisiting = st.number_input("People Visiting", 1, 10, 2)
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)

PreferredPropertyStar = st.selectbox("Hotel Rating", [3, 4, 5])
Passport = st.selectbox("Passport", [0, 1])
OwnCar = st.selectbox("Own Car", [0, 1])

PitchSatisfactionScore = st.slider("Pitch Satisfaction", 1, 5, 3)
NumberOfFollowups = st.slider("Followups", 0, 10, 2)
DurationOfPitch = st.slider("Pitch Duration", 5, 60, 20)

# =========================
# Input DataFrame
# =========================
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "ProductPitched": ProductPitched,
    "Designation": Designation,
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

    st.subheader("Prediction Result")
    st.success(result)

    st.write("Input Data")
    st.dataframe(input_data)

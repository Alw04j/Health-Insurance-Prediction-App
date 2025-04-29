import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from insurance_model import InsuranceModel, load_model, save_model, main

# Load the pre-trained insurance model
insurance_model = load_model()


# Set up the page with a custom title
st.set_page_config(page_title="Insurance Prediction App", page_icon="💼", layout="wide")

# Custom CSS for header and tab styling with increased tab font size
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(to right, #FFDEE9, #B5AAFF);
            color: #333333;
        }
        .css-1d391kg {
            font-size: 32px;
            text-align: center;
            color: #FF6F61;
            margin-bottom: 20px;
        }
        .css-1v3fvcr {
            padding: 20px;
        }
        .stTabs div[role="tablist"] {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
            font-size: 60px; /* Increased font size */
            color: #FF6F61;
        }
        .stTabs div[role="tablist"] button[role="tab"] {
            font-size: 60px; /* Increased font size */
            text-align: center;
            background: none;
            border: none;
            color: #FF6F61;
            cursor: pointer;
        }
        .stTabs div[role="tablist"] button[role="tab"]:hover {
            text-decoration: underline;
        }
        .stButton>button {
            background-color: #FF6F61;
            color: white;
            border-radius: 8px;
            padding: 12px;
            transition: background-color 0.3s ease, transform 0.3s ease;
            border: none;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #FF4F4F;
            transform: scale(1.05);
        }
        .stSelectbox, .stNumberInput, .stTextInput {
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)


# Display the app name above the tabs
st.markdown("<h1 style='text-align: center; color: #FF6F61;'>Insurance Prediction App</h1>", unsafe_allow_html=True)

# Tabs for navigation (Home, Prediction, Contribute) with increased font size
tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "Contribute"])

# Home Page (First Tab)
with tab1:
    st.title("Insurance Charges Prediction")
    st.write("""
        Welcome to the Insurance Charges Prediction app! This app predicts insurance charges based on various factors using a Gradient Boosting Regressor model.

        **Dataset Details:**
        - The dataset used is the **Insurance Dataset** from a public source.
        - It contains information about individuals, including their age, sex, BMI (Body Mass Index), number of children, smoking status, and region.
        - **Target Variable:** `charges` - the insurance charges for the individual.

        **Value Ranges:**
        - Age: 18 to 100 years
        - BMI: 10.0 to 50.0
        - Number of Children: 0 to 10
        - Smoking Status: Yes/No
        - Region: Northeast, Northwest, Southeast, Southwest

        The app uses this data to predict the insurance charges for an individual based on their profile.
    """)

    # Plot: Actual vs Predicted Charges (Using Plotly for better interaction)
    st.subheader("Actual vs Predicted Charges")
    fig1 = px.scatter(
        x=insurance_model.y_test, y=insurance_model.y_pred,
        labels={'x': 'Actual Charges', 'y': 'Predicted Charges'},
        title="Actual vs Predicted Charges",
        template="plotly_dark"
    )
    fig1.add_shape(
        type='line',
        line=dict(dash='dash', color='red'),
        x0=min(insurance_model.y_test), x1=max(insurance_model.y_test),
        y0=min(insurance_model.y_test), y1=max(insurance_model.y_test)
    )

    # Plot: Feature Importance (Using Plotly)
    st.subheader("Feature Importance")
    feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    importances = insurance_model.model.feature_importances_
    fig2 = go.Figure(go.Bar(
        x=importances,
        y=feature_names,
        orientation='h',
        marker=dict(color='royalblue'),
    ))
    fig2.update_layout(
        title="Feature Importance",
        xaxis_title="Relative Importance",
        yaxis_title="Features",
        template="plotly_dark"
    )

    # Display graphs side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # Display MSE and R² values
    st.subheader("Model Evaluation")
    st.write(f"**Mean Squared Error (MSE):** {insurance_model.mse:.2f}")
    st.write(f"**R-squared (R²):** {insurance_model.r2:.2f}")

# Prediction Page (Second Tab)
with tab2:
    st.title("Predict Insurance Charges")

    # Input fields for prediction with unique keys
    age = st.number_input("Age", min_value=18, max_value=100, value=30, key="age_pred")
    sex = st.selectbox("Sex", options=["male", "female"], key="sex_pred")
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, key="bmi_pred")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, key="children_pred")
    smoker = st.selectbox("Smoker", options=["yes", "no"], key="smoker_pred")
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"], key="region_pred")

    if st.button("Predict", key="predict_btn"):
        # Prediction
        predicted_charges = insurance_model.predict_insurance(age, sex, bmi, children, smoker, region)
        st.write(f"**Predicted Insurance Charges:** ${predicted_charges:.2f}")

# Contribute Page (Third Tab)
with tab3:
    st.title("Contribute Data")

    st.write("""
        **Add your data**: Contribute new entries to the model by providing the information below. 
        The new data will be added to the dataset, and the model will be retrained to improve its predictions.
    """)

    # Input fields for new data with unique keys
    age = st.number_input("Age", min_value=18, max_value=100, value=30, key="age_contrib")
    sex = st.selectbox("Sex", options=["male", "female"], key="sex_contrib")
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, key="bmi_contrib")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, key="children_contrib")
    smoker = st.selectbox("Smoker", options=["yes", "no"], key="smoker_contrib")
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"], key="region_contrib")
    charges = st.number_input("Charges", min_value=0.0, value=1000.0, key="charges_contrib")

    if st.button("Submit", key="submit_btn"):
        # Prepare new data for appending
        new_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region],
            'charges': [charges]
        })
        # Update dataset and retrain model
        insurance_model.append_data(new_data, 'health insurance.csv')
        # Save the updated model
        save_model(insurance_model)
        st.success("Data contributed successfully and model retrained!")

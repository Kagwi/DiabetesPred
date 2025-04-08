import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode Styling
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    .css-1v0mbdj.ebxwdo61 {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: #2E2E2E;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white;
    }
    .highlight {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #4A90E2;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Visualization: Feature Importance ---
def plot_feature_importance():
    features = ['HbA1c', 'Blood Glucose', 'Age', 'BMI', 'Hypertension', 
                'Heart Disease', 'Smoking History', 'Gender']
    importance = [0.643860, 0.317668, 0.021189, 0.009640, 0.004004, 
                  0.002767, 0.000554, 0.000319]

    plt.figure(figsize=(10, 6), facecolor='#1B1C20')
    ax = plt.gca()
    ax.set_facecolor('#1B1C20')
    
    bars = plt.barh(features, importance, color='#4A90E2')
    plt.title('Feature Importance in Prediction', color='white', pad=20)
    plt.xlabel('Importance Score', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1%}', 
                 ha='left', va='center', color='white',
                 fontweight='bold', fontsize=10,
                 bbox=dict(facecolor='#1B1C20', edgecolor='none', pad=5))
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.3, color='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    return plt

# --- Visualization: Risk Profiles ---
def plot_risk_profiles():
    profiles = ['Very Healthy', 'Moderately Healthy', 'Borderline', 
                'Slightly Elevated', 'High Risk', 'Very High Risk']
    hba1c = [4.5, 5.2, 5.7, 6.0, 7.0, 8.0]
    glucose = [80, 90, 100, 110, 160, 200]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1B1C20')
    fig.patch.set_facecolor('#1B1C20')

    ax1.set_facecolor('#1B1C20')
    ax1.plot(profiles, hba1c, marker='o', color='#4A90E2', linewidth=2)
    ax1.set_title('HbA1c Levels by Risk Profile', color='white', pad=20)
    ax1.set_ylabel('HbA1c Level', color='white')
    ax1.tick_params(axis='both', colors='white')
    ax1.grid(True, linestyle='--', alpha=0.3, color='white')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax2.set_facecolor('#1B1C20')
    ax2.plot(profiles, glucose, marker='o', color='#4A90E2', linewidth=2)
    ax2.set_title('Blood Glucose Levels by Risk Profile', color='white', pad=20)
    ax2.set_ylabel('Blood Glucose Level', color='white')
    ax2.tick_params(axis='both', colors='white')
    ax2.grid(True, linestyle='--', alpha=0.3, color='white')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('white')
    
    plt.tight_layout()
    return plt

# --- Main App ---
def main():
    st.title("Diabetes Risk Prediction System")
    st.markdown('<p class="header">Advanced Health Risk Assessment Tool</p>', unsafe_allow_html=True)

    # Disclaimer
    st.warning("""
        **MEDICAL DISCLAIMER**
        This tool provides an estimated risk assessment based on statistical analysis.
        It is not a substitute for professional medical diagnosis or advice.
        Please consult a qualified healthcare provider for proper evaluation.
    """)

    st.info("""
        This tool uses machine learning to assess diabetes risk based on health metrics.
        HbA1c and Blood Glucose levels are the most influential indicators.
    """)

    # --- Input Form ---
    st.subheader("Patient Information")
    st.markdown('<p class="header">Please fill in the following details:</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Personal Information**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        smoking_history = st.selectbox("Smoking History", 
                                       ["never", "current", "former", "ever", "No Info"])

    with col2:
        st.markdown("**Medical Metrics**")
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.0)
        blood_glucose = st.number_input("Blood Glucose Level", min_value=70, max_value=300, value=100)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    # --- Predict Button ---
    if st.button("Analyze Risk", help="Click to analyze diabetes risk based on provided information"):
        try:
            model = joblib.load("best_model.joblib")

            gender_encoded = 1 if gender == "Male" else 0
            hypertension_encoded = 1 if hypertension == "Yes" else 0
            heart_disease_encoded = 1 if heart_disease == "Yes" else 0
            smoking_map = {"never": 0, "current": 1, "former": 2, "ever": 3, "No Info": 4}
            smoking_encoded = smoking_map[smoking_history]

            # ✅ Correct Feature Order
            input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded,
                                    smoking_encoded, bmi, hba1c, blood_glucose]])

            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[0][1]

            st.subheader("Risk Assessment Results")
            st.markdown('<p class="header">Analysis Complete</p>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"

            with col1:
                st.metric("Risk Level", risk_level)
            with col2:
                st.metric("Risk Probability", f"{prediction_proba:.1%}")
            with col3:
                st.metric("Confidence", f"{max(prediction_proba, 1 - prediction_proba):.1%}")

            if prediction[0] == 1:
                st.error("""
                    ### High Risk Detected
                    - Consult with a healthcare provider
                    - Get a full diabetes screening
                    - Discuss lifestyle changes with your doctor
                """)
            else:
                st.success("""
                    ### Low Risk Detected
                    - Keep living healthy
                    - Schedule regular checkups
                    - Maintain a balanced lifestyle
                """)

            # --- Visualizations ---
            st.subheader("Risk Analysis Visualization")
            st.markdown('<p class="header">Data Insights</p>', unsafe_allow_html=True)
            st.markdown("#### Feature Importance")
            st.pyplot(plot_feature_importance())

            st.markdown("#### Risk Profiles Reference")
            st.pyplot(plot_risk_profiles())

            st.info("""
                **Understanding Your Results**
                - HbA1c Level (64.4%) is the top predictor
                - Blood Glucose (31.8%) follows closely
                - Other factors like age, BMI, and medical history play smaller roles
            """)

            st.warning("""
                **Important Notice**
                This model provides an estimate and should not replace professional diagnosis.
                Always consult your healthcare provider.
            """)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()

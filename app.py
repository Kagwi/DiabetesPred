import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        padding: 20px;
        border-radius: 10px;
    }
    .css-1v0mbdj.ebxwdo61 {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #1E88E5;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def plot_feature_importance():
    # Feature importance data
    features = ['HbA1c', 'Blood Glucose', 'Age', 'BMI', 'Hypertension', 
                'Heart Disease', 'Smoking History', 'Gender']
    importance = [0.643860, 0.317668, 0.021189, 0.009640, 0.004004, 
                 0.002767, 0.000554, 0.000319]
    
    # Create figure with dark background
    plt.figure(figsize=(10, 6), facecolor='#1B1C20')
    ax = plt.gca()
    ax.set_facecolor('#1B1C20')
    
    # Create horizontal bar plot
    bars = plt.barh(features, importance, color='#4A90E2')
    
    # Customize plot
    plt.title('Feature Importance in Prediction', color='white', pad=20)
    plt.xlabel('Importance Score', color='white')
    
    # Customize ticks
    plt.xticks(color='white')
    plt.yticks(color='white')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', 
                ha='left', va='center', color='white',
                fontweight='bold', fontsize=10,
                bbox=dict(facecolor='#1B1C20', edgecolor='none', pad=5))
    
    # Customize grid
    plt.grid(True, axis='x', linestyle='--', alpha=0.3, color='white')
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('white')
        
    plt.tight_layout()
    return plt

def plot_risk_profiles():
    # Sample data
    profiles = ['Very Healthy', 'Moderately Healthy', 'Borderline', 
                'Slightly Elevated', 'High Risk', 'Very High Risk']
    hba1c = [4.5, 5.2, 5.7, 6.0, 7.0, 8.0]
    glucose = [80, 90, 100, 110, 160, 200]
    
    # Create figure with dark background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1B1C20')
    fig.patch.set_facecolor('#1B1C20')
    
    # Plot HbA1c
    ax1.set_facecolor('#1B1C20')
    ax1.plot(profiles, hba1c, marker='o', color='#4A90E2', linewidth=2)
    ax1.set_title('HbA1c Levels by Risk Profile', color='white', pad=20)
    ax1.set_ylabel('HbA1c Level', color='white')
    ax1.tick_params(axis='both', colors='white')
    ax1.grid(True, linestyle='--', alpha=0.3, color='white')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot Glucose
    ax2.set_facecolor('#1B1C20')
    ax2.plot(profiles, glucose, marker='o', color='#4A90E2', linewidth=2)
    ax2.set_title('Blood Glucose Levels by Risk Profile', color='white', pad=20)
    ax2.set_ylabel('Blood Glucose Level', color='white')
    ax2.tick_params(axis='both', colors='white')
    ax2.grid(True, linestyle='--', alpha=0.3, color='white')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Customize spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('white')
    
    plt.tight_layout()
    return plt

def main():
    # Header with custom styling
    st.title("Diabetes Risk Prediction System")
    st.markdown('<p class="header">Advanced Health Risk Assessment Tool</p>', unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.warning("""
        **MEDICAL DISCLAIMER**
        
        This tool provides an estimated risk assessment based on statistical analysis. It is not a substitute 
        for professional medical diagnosis or advice. The predictions are based on a machine learning model 
        and should be used only as a screening tool. Please consult with a qualified healthcare provider 
        for proper medical evaluation and diagnosis.
    """)
    
    # Information about the tool
    st.info("""
        This tool uses machine learning to assess diabetes risk based on various health metrics. 
        The model considers multiple factors with different weights, with HbA1c and Blood Glucose 
        levels being the most significant indicators.
    """)
    
    # Input Section
    st.subheader("Patient Information")
    st.markdown('<p class="header">Please fill in the following details:</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personal Information**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0,
                            help="Body Mass Index (weight in kg / height in m²)")
        smoking_history = st.selectbox("Smoking History", 
            ["never", "current", "former", "ever", "No Info"])

    with col2:
        st.markdown("**Medical Metrics**")
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.0,
                               help="Glycated hemoglobin level (percentage)")
        blood_glucose = st.number_input("Blood Glucose Level", min_value=70, max_value=300, value=100,
                                      help="Fasting blood glucose level (mg/dL)")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    # Prediction Button
    if st.button("Analyze Risk", help="Click to analyze diabetes risk based on provided information"):
        try:
            model = joblib.load("best_model.joblib")
            
            # Data preprocessing
            gender_encoded = 1 if gender == "Male" else 0
            hypertension_encoded = 1 if hypertension == "Yes" else 0
            heart_disease_encoded = 1 if heart_disease == "Yes" else 0
            
            smoking_map = {
                "never": 0, "current": 1, "former": 2, "ever": 3, "No Info": 4
            }
            smoking_encoded = smoking_map[smoking_history]

            # Create feature array
            input_data = np.array([[
                gender_encoded, age, hypertension_encoded, heart_disease_encoded,
                smoking_encoded, bmi, hba1c, blood_glucose
            ]])

            # Get prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[0][1]

            # Display results
            st.subheader("Risk Assessment Results")
            st.markdown('<p class="header">Analysis Complete</p>', unsafe_allow_html=True)
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
            risk_color = "#FF4B4B" if prediction[0] == 1 else "#00CC96"
            
            with col1:
                st.metric("Risk Level", risk_level)
            with col2:
                st.metric("Risk Probability", f"{prediction_proba:.1%}")
            with col3:
                st.metric("Confidence", f"{max(prediction_proba, 1-prediction_proba):.1%}")

            # Risk interpretation
            if prediction[0] == 1:
                st.error("""
                    ### High Risk Detected
                    The model indicates an elevated risk of diabetes. It is strongly recommended to:
                    1. Consult with a healthcare provider for proper medical evaluation
                    2. Consider getting a comprehensive diabetes screening
                    3. Discuss lifestyle modifications with your doctor
                """)
            else:
                st.success("""
                    ### Low Risk Detected
                    The model indicates a lower risk of diabetes. Recommendations:
                    1. Continue maintaining a healthy lifestyle
                    2. Regular check-ups with your healthcare provider
                    3. Stay active and maintain a balanced diet
                """)

            # Visualizations
            st.subheader("Risk Analysis Visualization")
            st.markdown('<p class="header">Data Insights</p>', unsafe_allow_html=True)
            
            # Feature importance plot
            st.markdown("#### Feature Importance")
            st.pyplot(plot_feature_importance())
            
            # Risk profiles plot
            st.markdown("#### Risk Profiles Reference")
            st.pyplot(plot_risk_profiles())
            
            # Additional Information
            st.info("""
                **Understanding Your Results**
                
                The prediction is based on these key factors (in order of importance):
                1. HbA1c Level (64.4%) - Most significant indicator
                2. Blood Glucose (31.8%) - Second most important factor
                3. Age (2.1%) - Third most influential factor
                
                Other factors like BMI, hypertension, heart disease, smoking history, and gender 
                contribute to the overall prediction but have less impact on the final result.
            """)
            
            # Final Disclaimer
            st.warning("""
                **Important Notice**
                
                This risk assessment is based on a machine learning model and should not be used as the sole basis 
                for medical decisions. The model has limitations and may not account for all possible factors 
                affecting diabetes risk. Always consult with healthcare professionals for proper medical advice 
                and diagnosis.
            """)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
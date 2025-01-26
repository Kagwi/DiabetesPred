import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("C:/Users/NEONSOL/Desktop/Python Essentials.dcdb3279-00a6-4417-aa79-92b4fa893829/best_model.joblib")

def get_risk_level(hba1c, glucose):
    """Get clinical risk level based on medical guidelines"""
    if hba1c >= 6.5 or glucose >= 126:
        return "High"
    elif hba1c >= 5.7 or glucose >= 100:
        return "Moderate"
    else:
        return "Low"

def main():
    st.title("Diabetes Risk Prediction")
    st.write("Enter your health information for an AI-powered diabetes risk assessment")
    
    try:
        model = load_model()
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Primary Risk Factors")
            hba1c = st.number_input(
                "HbA1c Level (%)", 
                min_value=4.0, 
                max_value=9.0, 
                value=5.5,
                help="Normal: <5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥6.5%"
            )
            
            glucose = st.number_input(
                "Blood Glucose Level (mg/dL)", 
                min_value=70, 
                max_value=300, 
                value=100,
                help="Normal: <100, Prediabetes: 100-125, Diabetes: ≥126"
            )
            
            bmi = st.number_input(
                "BMI", 
                min_value=15.0, 
                max_value=50.0, 
                value=25.0,
                help="Normal: <25, Overweight: 25-29.9, Obese: ≥30"
            )
            
            age = st.number_input(
                "Age", 
                min_value=18, 
                max_value=100, 
                value=30,
                help="Age in years"
            )
        
        with col2:
            st.subheader("Additional Factors")
            gender = st.radio(
                "Gender",
                options=["Female", "Male"],
                horizontal=True,
                help="Select biological gender"
            )
            
            hypertension = st.radio(
                "Hypertension",
                options=["No", "Yes"],
                horizontal=True,
                help="Do you have high blood pressure?"
            )
            
            heart_disease = st.radio(
                "Heart Disease",
                options=["No", "Yes"],
                horizontal=True,
                help="Do you have any heart disease?"
            )
            
            smoking_history = st.selectbox(
                "Smoking History",
                options=["never", "former", "current", "not current", "ever", "No Info"],
                help="Select your smoking status"
            )
        
        if st.button("Predict Risk", use_container_width=True):
            # Prepare features for the model
            features = pd.DataFrame([[
                1 if gender == "Male" else 0,
                age,
                1 if hypertension == "Yes" else 0,
                1 if heart_disease == "Yes" else 0,
                {"never": 0, "former": 1, "current": 2, 
                 "not current": 3, "ever": 4, "No Info": 5}[smoking_history],
                bmi,
                hba1c,
                glucose
            ]], columns=['gender', 'age', 'hypertension', 'heart_disease', 
                        'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
            
            # Get model prediction and clinical risk level
            prediction = model.predict(features)[0]
            clinical_risk = get_risk_level(hba1c, glucose)
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("AI Model Prediction")
                if prediction == 1:
                    st.warning("Higher Risk Indicated")
                    st.markdown("""
                    **Recommended Actions:**
                    - Consult with a healthcare provider
                    - Get comprehensive diabetes screening
                    - Review lifestyle factors
                    """)
                else:
                    st.success("Lower Risk Indicated")
                    st.markdown("""
                    **Recommendations:**
                    - Maintain healthy lifestyle
                    - Continue regular check-ups
                    - Monitor for changes
                    """)
            
            with col2:
                st.subheader("Clinical Guidelines")
                if clinical_risk == "High":
                    st.error("Clinical High Risk")
                    st.write("Your HbA1c or glucose levels are in the high-risk range")
                elif clinical_risk == "Moderate":
                    st.warning("Clinical Moderate Risk")
                    st.write("Your levels indicate pre-diabetes range")
                else:
                    st.success("Clinical Low Risk")
                    st.write("Your levels are within normal range")
            
            # Display key metrics
            st.markdown("---")
            st.subheader("Key Metrics Analysis")
            
            # Create three columns for metrics
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    "HbA1c",
                    f"{hba1c}%",
                    delta=f"{hba1c - 5.7:.1f}% from threshold",
                    delta_color="inverse"
                )
            
            with m2:
                st.metric(
                    "Blood Glucose",
                    f"{glucose} mg/dL",
                    delta=f"{glucose - 100:.0f} from threshold",
                    delta_color="inverse"
                )
            
            with m3:
                st.metric(
                    "BMI",
                    f"{bmi:.1f}",
                    delta=f"{bmi - 25:.1f} from normal range",
                    delta_color="inverse"
                )
            
            # Display feature importance
            st.markdown("---")
            st.subheader("Risk Factors Analysis")
            
            importance_values = [0.029, 2.224, 0.405, 0.263, 0.055, 0.949, 64.059, 32.016]
            importance_df = pd.DataFrame({
                "Factor": ["Gender", "Age", "Hypertension", "Heart Disease", 
                          "Smoking History", "BMI", "HbA1c Level", "Blood Glucose Level"],
                "Importance (%)": importance_values
            })
            importance_df = importance_df.sort_values("Importance (%)", ascending=True)
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(importance_df["Factor"], importance_df["Importance (%)"])
            
            # Customize chart
            ax.set_xlabel("Importance (%)")
            ax.set_title("Feature Importance in Risk Prediction")
            
            # Add value labels on the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f"{width:.1f}%",
                       ha='left', va='center')
            
            st.pyplot(fig)
            
            # Additional information
            st.markdown("---")
            st.subheader("Understanding Your Results")
            st.info("""
            This assessment combines:
            1. AI Model Prediction based on all provided factors
            2. Clinical Guidelines based on standard medical thresholds
            
            **Key Thresholds:**
            - HbA1c: Normal < 5.7%, Pre-diabetes 5.7-6.4%, Diabetes ≥ 6.5%
            - Blood Glucose: Normal < 100, Pre-diabetes 100-125, Diabetes ≥ 126 mg/dL
            - BMI: Normal < 25, Overweight 25-29.9, Obese ≥ 30
            
            *This tool provides risk assessment based on available data but should not be considered a diagnosis. 
            Always consult healthcare professionals for proper medical evaluation.*
            """)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please ensure all inputs are valid.")

if __name__ == "__main__":
    main()

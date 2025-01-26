import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stTitle {
        font-size: 32px;
        color: #4A90E2;
        font-weight: bold;
    }
    .stSubheader {
        color: #4A90E2;
        font-weight: 600;
    }
    .header {
        font-size: 24px;
        color: #4A90E2;
        font-weight: bold;
    }
    .card {
        background-color: #1F1F1F;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .stMetric {
        color: #76E1A6;
    }
    .stSelectbox, .stNumberInput {
        background-color: #2C2C2C;
        border: 1px solid #4A90E2;
        color: white;
    }
    .stSelectbox select {
        background-color: #2C2C2C;
    }
    </style>
""", unsafe_allow_html=True)

def plot_feature_importance():
    # Feature importance data
    features = ['HbA1c', 'Blood Glucose', 'Age', 'BMI', 'Hypertension', 
                'Heart Disease', 'Smoking History', 'Gender']
    importance = [0.643860, 0.317668, 0.021189, 0.009640, 0.004004, 
                  0.002767, 0.000554, 0.000319]

    # Plot
    plt.figure(figsize=(10, 6), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    
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
                 bbox=dict(facecolor='#121212', edgecolor='none', pad=5))
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.3, color='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        
    plt.tight_layout()
    return plt

def main():
    st.title("Diabetes Risk Prediction System")
    st.markdown('<p class="header">Advanced Health Risk Assessment Tool</p>', unsafe_allow_html=True)

    # Add medical disclaimer
    st.markdown("""
        <div class="card">
        <p style="color: #FF4B4B; font-weight: bold;">**MEDICAL DISCLAIMER**</p>
        <p>This tool is for informational purposes only. Please consult a healthcare professional for a full diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)

    st.info("This tool uses machine learning to assess diabetes risk based on health metrics. Fill in the details below to analyze your risk.")

    # User inputs (Split into two columns for better layout)
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "ever", "No Info"])

    with col2:
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.0)
        blood_glucose = st.number_input("Blood Glucose Level", min_value=70, max_value=300, value=100)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    # Analyze button
    analyze_button = st.button("Analyze Risk")
    if analyze_button:
        with st.spinner('Analyzing...'):
            try:
                model = joblib.load("best_model.joblib")

                gender_encoded = 1 if gender == "Male" else 0
                hypertension_encoded = 1 if hypertension == "Yes" else 0
                heart_disease_encoded = 1 if heart_disease == "Yes" else 0
                smoking_map = {"never": 0, "current": 1, "former": 2, "ever": 3, "No Info": 4}
                smoking_encoded = smoking_map[smoking_history]

                # Prepare input data
                input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
                                        smoking_encoded, bmi, hba1c, blood_glucose]])

                # Predict risk and probability
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[0][1]

                risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
                st.metric("Risk Level", risk_level)
                st.metric("Risk Probability", f"{prediction_proba:.1%}")

                # Display feature importance chart
                st.subheader("Feature Importance Visualization")
                st.pyplot(plot_feature_importance())

                # Display conclusion message
                if prediction[0] == 1:
                    st.warning("**High Risk**: It is recommended that you consult with a healthcare professional.")
                else:
                    st.success("**Low Risk**: Keep up with a healthy lifestyle!")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

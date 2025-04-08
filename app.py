import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Configure layout
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global dark theme styling
st.markdown("""
    <style>
    body, .main, .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    .stAlert, .css-1v0mbdj {
        background-color: #1f1f1f;
        color: #ffffff;
        border-left: 5px solid #4A90E2;
    }
    .header {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 10px;
    }
    label, .css-1cpxqw2, .css-10trblm {
        color: #ffffff !important;
    }
    .stTextInput > div > div > input {
        color: white !important;
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Plotting functions 
def plot_feature_importance():
    features = ['HbA1c', 'Blood Glucose', 'Age', 'BMI', 'Hypertension',
                'Heart Disease', 'Smoking History', 'Gender']
    importance = [0.643860, 0.317668, 0.021189, 0.009640, 0.004004,
                  0.002767, 0.000554, 0.000319]

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
                 f'{width:.1%}', ha='left', va='center', color='white',
                 fontweight='bold', fontsize=10,
                 bbox=dict(facecolor='#121212', edgecolor='none', pad=5))

    plt.grid(True, axis='x', linestyle='--', alpha=0.3, color='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    return plt

def plot_risk_profiles():
    profiles = ['Very Healthy', 'Moderately Healthy', 'Borderline',
                'Slightly Elevated', 'High Risk', 'Very High Risk']
    hba1c = [4.5, 5.2, 5.7, 6.0, 7.0, 8.0]
    glucose = [80, 90, 100, 110, 160, 200]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#121212')
    fig.patch.set_facecolor('#121212')

    for ax, y, title, ylabel in zip([ax1, ax2], [hba1c, glucose],
                                    ['HbA1c Levels by Risk Profile', 'Blood Glucose Levels by Risk Profile'],
                                    ['HbA1c Level', 'Blood Glucose Level']):
        ax.set_facecolor('#121212')
        ax.plot(profiles, y, marker='o', color='#4A90E2', linewidth=2)
        ax.set_title(title, color='white', pad=20)
        ax.set_ylabel(ylabel, color='white')
        ax.tick_params(axis='both', colors='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return plt

# Main app logic 
def main():
    st.title("🩺 Diabetes Risk Prediction")

    st.warning("This tool provides a preliminary risk estimate. Please consult a licensed doctor for medical advice.")

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 0, 120, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "No Info"])
    with col2:
        hba1c = st.number_input("HbA1c (%)", 3.0, 10.0, 5.5)
        glucose = st.number_input("Glucose (mg/dL)", 70, 300, 100)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    if st.button("Analyze Risk"):
        try:
            model = joblib.load("best_model.joblib")
            gender_val = 1 if gender == "Male" else 0
            hypertension_val = 1 if hypertension == "Yes" else 0
            heart_val = 1 if heart_disease == "Yes" else 0
            smoking_map = {"never": 0, "current": 1, "former": 2, "ever": 3, "No Info": 4}
            smoke_val = smoking_map[smoking]

            input_data = np.array([[gender_val, age, hypertension_val,
                                    heart_val, smoke_val, bmi, hba1c, glucose]])

            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[0][1]

            risk_label = "High Risk" if prediction[0] == 1 else "Low Risk"
            risk_color = "#FF4B4B" if prediction[0] == 1 else "#00CC96"

            st.subheader("🧪 Risk Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Level", risk_label)
            col2.metric("Probability", f"{prediction_proba:.1%}")
            col3.metric("Confidence", f"{max(prediction_proba, 1 - prediction_proba):.1%}")

            if prediction[0] == 1:
                st.error("⚠️ High Risk Detected. Please seek medical attention immediately.")
            else:
                st.success("✅ Low Risk. Keep maintaining healthy habits.")

            st.markdown("### 🔍 Feature Importance")
            st.pyplot(plot_feature_importance())

            st.markdown("### 📈 Risk Profiles Reference")
            st.pyplot(plot_risk_profiles())

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    main()

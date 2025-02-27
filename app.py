import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
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
    </style>
""", unsafe_allow_html=True)


def load_model():
    """Load the trained attrition prediction model."""
    try:
        return joblib.load("best_model_random_forest.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def plot_feature_importance(model):
    """Plot feature importance for the model."""
    if hasattr(model, 'feature_importances_'):
        features = ['Age', 'Job Level', 'Total Working Years', 'Years at Company', 'Overtime', 'Job Satisfaction']
        importance = model.feature_importances_

        cmap = ListedColormap(sns.color_palette("coolwarm", len(features)).as_hex())

        plt.figure(figsize=(8, 6), facecolor='#121212')
        ax = plt.gca()
        ax.set_facecolor('#121212')

        bars = plt.bar(features, importance, color=cmap.colors, edgecolor='white', linewidth=0.7)
        plt.title('Feature Importance in Prediction', color='white', pad=20, fontsize=16, fontweight='bold')
        plt.ylabel('Importance Score', color='white', fontsize=12)
        plt.xticks(color='white', fontsize=10, rotation=45)
        plt.yticks(color='white', fontsize=10)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.1%}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

        plt.grid(True, axis='y', linestyle='--', alpha=0.3, color='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        plt.tight_layout()
        return plt
    else:
        st.warning("Feature importance data not available.")


def display_hr_disclaimer():
    """Display a disclaimer about the limitations of the prediction model."""
    st.markdown("""
        <div class="card">
        <h3 style="color: #FF4B4B;">HR Disclaimer</h3>
        <p style="color: white; font-size: 14px;">
        This tool provides a prediction for employee attrition risk based on available data.
        It should not be used as the sole determinant for HR decisions. For best results,
        consult HR professionals and consider other qualitative factors.
        </p>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.title("Employee Attrition Predictor")
    st.markdown('<p class="header">AI-Powered Employee Retention Insights</p>', unsafe_allow_html=True)

    # Load the model
    model = load_model()
    if model is None:
        return

    # Display disclaimer
    display_hr_disclaimer()

    st.info("This tool predicts employee attrition risk based on job-related metrics. Fill in the details below to analyze attrition risk.")

    # User inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)

    with col2:
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        job_satisfaction = st.selectbox("Job Satisfaction (1-Low to 4-High)", [1, 2, 3, 4])

    if st.button("Analyze Risk"):
        with st.spinner('Analyzing...'):
            try:
                # Encode categorical variables
                overtime_encoded = 1 if overtime == "Yes" else 0

                input_data = np.array([[age, job_level, total_working_years, years_at_company, overtime_encoded, job_satisfaction]])

                # Prediction
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[0][1]

                risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
                st.metric("Risk Level", risk_level)
                st.metric("Attrition Probability", f"{prediction_proba:.1%}")

                # Recommendations
                st.subheader("Retention Strategies")
                if prediction[0] == 1:
                    st.warning("""
                        - Conduct one-on-one employee engagement meetings.
                        - Provide opportunities for professional growth and training.
                        - Offer flexible working arrangements to improve work-life balance.
                        - Address job satisfaction concerns and career progression paths.
                        - Implement recognition programs to appreciate employees' contributions.
                    """)
                else:
                    st.success("""
                        - Continue fostering a positive work environment.
                        - Maintain an open feedback system with employees.
                        - Encourage mentorship and leadership development programs.
                        - Regularly review compensation and benefits.
                    """)

                # Display feature importance
                st.subheader("Feature Importance Visualization")
                st.pyplot(plot_feature_importance(model))

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

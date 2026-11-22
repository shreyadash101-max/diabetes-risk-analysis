
import streamlit as st
import pickle
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# -------------------------------------------------
# PROJECT TITLE
# -------------------------------------------------

st.title("🩺 Explainable AI-Based Diabetes Risk Prediction")

st.markdown(
    """
    ### Welcome to the Diabetes Risk Prediction Dashboard

    This project uses Machine Learning algorithms to predict
    diabetes classification based on health-related indicators.

    It also incorporates Explainable AI techniques to understand
    how different features influence model predictions.

    The dashboard provides diabetes prediction, risk factor analysis,
    model explainability, model evaluation, and algorithm comparison.
    """
)

st.divider()


# -------------------------------------------------
# LOAD TRAINED RANDOM FOREST MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():

    model_path = BASE_DIR / "models" / "diabetes_model.pkl"

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


try:

    model = load_model()

    st.success(
        "Trained Random Forest model loaded successfully!"
    )

except Exception as e:

    st.error(f"Model loading error: {e}")

    model = None


# -------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset Records",
    "251"
)

col2.metric(
    "Input Features",
    "16"
)

col3.metric(
    "Algorithms Compared",
    "9"
)

st.divider()


# -------------------------------------------------
# PROJECT OBJECTIVES
# -------------------------------------------------

st.header("Project Objectives")

st.markdown(
    """
    - Predict diabetes classification using Machine Learning.
    - Compare nine classification algorithms.
    - Evaluate model performance using three train-test split ratios.
    - Perform 5-fold Stratified Cross-Validation.
    - Identify important predictive health indicators.
    - Explain model behaviour using SHAP.
    - Display model performance through an interactive dashboard.
    """
)

st.divider()


# -------------------------------------------------
# TECHNOLOGIES USED
# -------------------------------------------------

st.header("Technologies Used")

st.markdown(
    """
    | Technology | Purpose |
    |---|---|
    | Python | Programming language |
    | Pandas | Data manipulation |
    | NumPy | Numerical operations |
    | Scikit-learn | Machine Learning and evaluation |
    | XGBoost | Gradient boosting |
    | LightGBM | Gradient boosting |
    | CatBoost | Gradient boosting |
    | SHAP | Model explainability |
    | Plotly | Interactive visualizations |
    | Streamlit | Dashboard development |
    """
)

st.divider()


# -------------------------------------------------
# SELECTED MODEL PERFORMANCE
# -------------------------------------------------

st.header("Selected Model Performance")

st.write(
    "The following values represent the recorded test-set "
    "performance of the selected Random Forest model."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Test Accuracy",
    "94.12%"
)

col2.metric(
    "Test F1 Score",
    "95.65%"
)

col3.metric(
    "ROC-AUC",
    "96.61%"
)

st.caption(
    "These are Random Forest test-set results, not the "
    "five-fold cross-validation averages."
)

st.divider()


# -------------------------------------------------
# MODEL EVALUATION OVERVIEW
# -------------------------------------------------

st.header("Model Evaluation Overview")

st.markdown(
    """
    This project uses two complementary evaluation approaches:

    **1. Cross-Validation Evaluation**
    - Four classification algorithms.
    - 5-fold Stratified Cross-Validation.
    - Accuracy, Precision, F1-score, AUPR,
      Average Precision, Balanced Accuracy, and ROC-AUC.

    **2. Model Comparison**
    - Nine machine learning algorithms/configurations.
    - Three train-test split ratios: 80:20, 70:30, and 90:10.
    - Comparison using multiple performance metrics
      and confusion matrices.
    """
)

st.divider()


# -------------------------------------------------
# DASHBOARD NAVIGATION
# -------------------------------------------------

st.header("Explore the Dashboard")

st.markdown(
    """
    Use the navigation menu in the sidebar to explore the
    different sections of the application:

    1. **Prediction:** Enter health indicators and obtain a diabetes prediction.
    2. **Risk Factor Analysis:** Explore important predictive features.
    3. **SHAP Explainability:** Understand feature contributions to model predictions.
    4. **Model Evaluation:** Examine 5-fold cross-validation results and compare four algorithms.
    5. **Model Comparison:** Compare nine machine learning algorithms across three train-test splits.
    """
)

st.divider()


# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------

st.warning(
    "Disclaimer: This application is an educational machine "
    "learning prototype. It is not intended for medical diagnosis "
    "or treatment."
)

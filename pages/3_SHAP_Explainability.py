
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SHAP Explainability",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
SHAP_DIR = BASE_DIR / "shap_exports"

st.title("Explainable AI - SHAP Analysis")

st.write(
    "SHAP (SHapley Additive exPlanations) helps explain "
    "how input features contribute to machine learning predictions."
)

st.success(
    "Model being explained: Random Forest"
)

st.info(
    "This page presents SHAP explanations for the trained "
    "Random Forest model only. The other three algorithms "
    "are compared on the Prediction and Model Evaluation pages."
)

st.write(
    "The grouped charts combine one-hot encoded categories "
    "to show the contribution of the original 16 input features."
)

# 1. Grouped feature importance
st.header("1. Grouped SHAP Feature Importance")

st.write(
    "This chart ranks the 16 original input features using "
    "their mean absolute grouped SHAP contributions. "
    "The encoded categories of each feature are combined."
)

importance_path = SHAP_DIR / "shap_grouped_importance.png"

if importance_path.exists():
    st.image(
        str(importance_path),
        caption="Random Forest - SHAP Importance (16 Original Features)",
        use_container_width=True
    )
else:
    st.error("Grouped SHAP importance chart not found.")

# 2. Grouped contribution distribution
st.header("2. Grouped SHAP Contribution Distribution")

st.write(
    "Each dot represents one observation. Its horizontal "
    "position indicates the grouped SHAP contribution "
    "of that feature for that observation."
)

st.markdown("""
- **Positive contribution:** Pushes the explained model output higher.
- **Negative contribution:** Pushes the explained model output lower.
- **Near zero:** Has little contribution for that observation.
""")

contribution_path = SHAP_DIR / "shap_grouped_contributions.png"

if contribution_path.exists():
    st.image(
        str(contribution_path),
        caption="Random Forest - Grouped SHAP Contributions",
        use_container_width=True
    )
else:
    st.error("Grouped SHAP contribution chart not found.")

# 3. Original encoded-feature explanations
st.header("3. Detailed One-Hot Encoded SHAP Analysis")

st.write(
    "These charts show individual encoded categories. "
    "For example, Gender_Female and Gender_Male appear separately."
)

original_importance_path = SHAP_DIR / "shap_feature_importance.png"

if original_importance_path.exists():
    st.image(
        str(original_importance_path),
        caption="Random Forest - SHAP Importance (31 Encoded Features)",
        use_container_width=True
    )
else:
    st.error("Original SHAP feature importance chart not found.")

original_summary_path = SHAP_DIR / "shap_summary.png"

if original_summary_path.exists():
    st.image(
        str(original_summary_path),
        caption="Random Forest - SHAP Summary Plot (31 Encoded Features)",
        use_container_width=True
    )
else:
    st.error("Original SHAP summary plot not found.")

st.warning(
    "SHAP explanations describe the behaviour of the Random Forest "
    "model. They do not establish medical causation or provide a "
    "clinically validated diabetes risk assessment."
)


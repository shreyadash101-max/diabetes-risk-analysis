
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

# Locate Excel workbook
BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_PATH = (
    BASE_DIR
    / "model_comparison"
    / "results"
    / "Diabetes_9_Model_Comparison_Results.xlsx"
)

st.title("Machine Learning Model Comparison")

st.write(
    "Comparison of nine machine learning algorithms "
    "across three train-test split ratios."
)

# Load workbook
@st.cache_data
def load_results():
    return {
        "80-20": pd.read_excel(
            EXCEL_PATH,
            sheet_name="80-20 Results"
        ),
        "70-30": pd.read_excel(
            EXCEL_PATH,
            sheet_name="70-30 Results"
        ),
        "90-10": pd.read_excel(
            EXCEL_PATH,
            sheet_name="90-10 Results"
        ),
        "Overall": pd.read_excel(
            EXCEL_PATH,
            sheet_name="Overall Comparison"
        ),
        "Confusion": pd.read_excel(
            EXCEL_PATH,
            sheet_name="Confusion Matrices"
        )
    }


if not EXCEL_PATH.exists():
    st.error("Excel results file not found. Check the file path.")
    st.stop()

data = load_results()

# Select split
split = st.selectbox(
    "Select Train-Test Split",
    ["80-20", "70-30", "90-10"]
)

df = data[split]

st.subheader(f"{split} Split Results")

# Display metrics
metric_columns = [
    "Accuracy",
    "Precision",
    "F1 Score",
    "AUPR",
    "Average Precision (AP)",
    "Balanced Accuracy",
    "AUC (ROC)"
]

st.dataframe(
    df[["Model"] + metric_columns].style.format(
        {col: "{:.4f}" for col in metric_columns}
    ),
    use_container_width=True
)

# F1 comparison chart
st.subheader("F1 Score Comparison")

fig = px.bar(
    df,
    x="Model",
    y="F1 Score",
    color="Model",
    title=f"Model F1 Scores - {split} Split",
    text_auto=".3f"
)

fig.update_layout(
    xaxis_tickangle=-35,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Overall comparison
st.subheader("Overall Model Performance")

overall = data["Overall"]

st.dataframe(
    overall.style.format(
        {col: "{:.4f}" for col in metric_columns}
    ),
    use_container_width=True
)

# Confusion matrices
st.subheader("Confusion Matrix Values")

confusion = data["Confusion"]

selected_model = st.selectbox(
    "Select Model",
    df["Model"].tolist()
)

cm_row = confusion[
    (confusion["Split"] == split)
    & (confusion["Model"] == selected_model)
].iloc[0]

cm = [
    [cm_row["TN"], cm_row["FP"]],
    [cm_row["FN"], cm_row["TP"]]
]

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["Predicted Negative", "Predicted Positive"],
    y=["Actual Negative", "Actual Positive"],
    color_continuous_scale="Blues",
    title=f"{selected_model} - {split} Confusion Matrix"
)

st.plotly_chart(fig_cm, use_container_width=True)

st.info(
    "These results are based on individual stratified "
    "train-test splits. Performance may vary with different "
    "random splits, especially with a small dataset."
)

import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title(" Classification Model Evaluation")

st.write(
    "Evaluate and compare the performance of different "
    "classification algorithms used for diabetes risk prediction."
)


# -------------------------------------------------
# 1. CROSS-VALIDATION RESULTS
# -------------------------------------------------

cv_results = pd.DataFrame({

    "Algorithm": [
        "Random Forest",
        "SVM",
        "Gradient Boosting",
        "Logistic Regression"
    ],

    "Mean Accuracy (%)": [
        93.50, 93.50, 91.00, 90.00
    ],

    "Std Accuracy (%)": [
        3.39, 3.00, 1.22, 3.16
    ],

    "Mean Precision (%)": [
        95.72, 96.37, 92.45, 96.11
    ],

    "Std Precision (%)": [
        2.56, 2.26, 3.02, 2.44
    ],

    "Mean F1-score (%)": [
        95.24, 95.23, 93.55, 92.42
    ],

    "Std F1-score (%)": [
        2.60, 2.18, 1.01, 2.51
    ],

    "Mean AUPR (%)": [
        98.61, 98.92, 98.47, 98.63
    ],

    "Std AUPR (%)": [
        0.69, 0.35, 0.53, 0.51
    ],

    "Mean Average Precision (%)": [
        98.58, 98.94, 98.50, 98.65
    ],

    "Std Average Precision (%)": [
        0.73, 0.34, 0.52, 0.50
    ],

    "Mean Balanced Accuracy (%)": [
        92.60, 93.00, 88.54, 90.58
    ],

    "Std Balanced Accuracy (%)": [
        3.28, 3.19, 2.49, 3.17
    ],

    "Mean ROC-AUC (%)": [
        97.10, 97.66, 96.85, 97.04
    ],

    "Std ROC-AUC (%)": [
        1.33, 0.69, 0.86, 0.96
    ]
})


# -------------------------------------------------
# 2. SELECT COMPARISON METRIC
# -------------------------------------------------

st.subheader("Algorithm Performance")

metric_options = [
    "Mean Accuracy (%)",
    "Mean Precision (%)",
    "Mean F1-score (%)",
    "Mean AUPR (%)",
    "Mean Average Precision (%)",
    "Mean Balanced Accuracy (%)",
    "Mean ROC-AUC (%)"
]

metric = st.selectbox(
    "Select metric for comparison",
    metric_options,
    index=2
)


# -------------------------------------------------
# 3. IDENTIFY BEST-PERFORMING ALGORITHM
# -------------------------------------------------

best_index = cv_results[metric].idxmax()

best_algorithm = cv_results.loc[
    best_index, "Algorithm"
]

best_score = cv_results.loc[
    best_index, metric
]


# -------------------------------------------------
# 4. DISPLAY BEST MODEL RESULT
# -------------------------------------------------

st.success(
    f"Best Performing Algorithm for {metric}: "
    f"{best_algorithm}"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Best Algorithm",
        value=best_algorithm
    )

with col2:

    st.metric(
        label=metric,
        value=f"{best_score:.2f}%"
    )

st.info(
    f"{best_algorithm} achieved the highest "
    f"{metric} of {best_score:.2f}% "
    f"across the evaluated algorithms."
)


# -------------------------------------------------
# 5. CROSS-VALIDATION RESULTS TABLE
# -------------------------------------------------

st.subheader("Cross-Validation Results Table")

display_results = cv_results.copy()

display_results = display_results.sort_values(
    by="Mean F1-score (%)",
    ascending=False
)

percentage_columns = [
    col for col in display_results.columns
    if col != "Algorithm"
]

st.dataframe(
    display_results.style.format({
        col: "{:.2f}%"
        for col in percentage_columns
    }),
    use_container_width=True,
    hide_index=True
)


# -------------------------------------------------
# 6. MODEL PERFORMANCE COMPARISON CHART
# -------------------------------------------------

st.subheader("Model Performance Comparison")

std_metric = metric.replace(
    "Mean ",
    "Std "
)

fig = px.bar(
    cv_results.sort_values(
        by=metric,
        ascending=False
    ),
    x="Algorithm",
    y=metric,
    text=metric,
    error_y=std_metric,
    title=f"5-Fold Cross-Validation - {metric}",
    labels={
        metric: metric,
        "Algorithm": "Classification Algorithm"
    }
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    yaxis_range=[0, 110],
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------------------------
# 7. RANDOM FOREST TEST-SET RESULTS
# -------------------------------------------------

st.divider()

st.subheader("Selected Model: Random Forest")

st.write(
    "The following metrics represent the recorded test-set "
    "evaluation of the selected Random Forest model."
)

test_metrics = {
    "Accuracy": 94.12,
    "Precision": 97.06,
    "Recall": 94.29,
    "F1-score": 95.65,
    "ROC-AUC": 96.61
}

metric_cols = st.columns(len(test_metrics))

for col, (name, value) in zip(
    metric_cols,
    test_metrics.items()
):

    with col:

        st.metric(
            label=name,
            value=f"{value:.2f}%"
        )


# -------------------------------------------------
# 8. RANDOM FOREST CONFUSION MATRIX
# -------------------------------------------------

st.subheader("Random Forest Confusion Matrix")

confusion_matrix = pd.DataFrame(
    [[15, 1], [2, 33]],
    index=[
        "Actual Negative",
        "Actual Positive"
    ],
    columns=[
        "Predicted Negative",
        "Predicted Positive"
    ]
)

st.dataframe(
    confusion_matrix,
    use_container_width=True
)

st.caption(
    "The confusion matrix is based on 51 test samples. "
    "These test-set metrics are reported for Random Forest; "
    "the comparison above uses 5-fold cross-validation F1-scores "
    "and other evaluation metrics."
)


# -------------------------------------------------
# 9. MODEL SELECTION INTERPRETATION
# -------------------------------------------------

st.subheader("Model Selection Interpretation")

st.write(
    "Random Forest achieved the highest mean cross-validation "
    "F1-score of 95.24% among the four evaluated algorithms. "
    "SVM achieved a very similar F1-score of 95.23%. "
    "Therefore, the difference between the two models is very small."
)

st.write(
    "SVM achieved the highest mean precision, AUPR, "
    "Average Precision, Balanced Accuracy, and ROC-AUC "
    "among the four evaluated algorithms."
)

st.warning(
    "This project is an educational prototype. Its results are "
    "based on a small dataset and are not intended for medical diagnosis."
)

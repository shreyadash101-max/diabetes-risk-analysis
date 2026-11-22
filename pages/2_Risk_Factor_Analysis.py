
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Risk Factor Analysis",
    layout="wide"
)

# Page heading
st.title("Diabetes Risk Factor Analysis")

st.write(
    "Explore the relationship between different health factors "
    "and diabetes cases in the dataset."
)

st.info(
    "These charts show associations in the dataset and model-specific "
    "predictive importance. They do not establish causation or provide "
    "medical diagnoses."
)


# Load files
@st.cache_data
def load_data():
    data = pd.read_csv("data/diabetes_data_upload.csv")
    importance = pd.read_csv("models/feature_importance.csv")

    data.columns = data.columns.str.strip()
    importance.columns = importance.columns.str.strip()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    return data, importance


try:
    df, importance_df = load_data()

    # Identify feature and importance columns
    feature_col = next(
        (c for c in importance_df.columns if c.lower() == "feature"),
        None
    )

    value_col = next(
        (c for c in importance_df.columns if c.lower() == "importance"),
        None
    )

    if feature_col is None or value_col is None:
        st.error(
            "Could not identify Feature and Importance columns "
            "in feature_importance.csv."
        )
        st.stop()

    # Identify target column
    target_col = next(
        (c for c in df.columns if c.lower() == "class"),
        None
    )

    if target_col is None:
        st.error("Target column 'class' was not found.")
        st.stop()

    # Convert target into numeric values
    target_values = pd.to_numeric(df[target_col], errors="coerce")

    # Dataset overview
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", 16)

    positive_count = int((target_values == 1).sum())

    col3.metric("Positive Cases", positive_count)

    st.divider()

    # Random Forest permutation feature importance
    st.header("Permutation Feature Importance")

    st.write(
        "Permutation feature importance measures how much model "
        "performance decreases when the values of an individual "
        "feature are randomly shuffled."
    )

    st.caption(
        "This analysis uses the trained Random Forest model. "
        "Higher values indicate greater predictive importance, "
        "not that a feature causes diabetes."
    )

    # Convert importance values to numeric
    importance_df[value_col] = pd.to_numeric(
        importance_df[value_col],
        errors="coerce"
    )

    importance_df = importance_df.dropna(
        subset=[value_col, feature_col]
    ).sort_values(
        value_col,
        ascending=False
    )

    if importance_df.empty:
        st.warning("No valid feature importance values were found.")

    else:
        fig = px.bar(
            importance_df,
            x=value_col,
            y=feature_col,
            orientation="h",
            title="Random Forest - Permutation Feature Importance",
            labels={
                value_col: "Decrease in F1 Score",
                feature_col: "Feature"
            },
            hover_data={value_col: ":.4f"}
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=650,
            xaxis_title="Decrease in F1 Score",
            yaxis_title="Feature",
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # Dataset-based risk factor analysis
    st.header("Dataset-Based Risk Factor Analysis")

    st.write(
        "Explore how the percentage of positive diabetes cases "
        "varies across different health factors."
    )

    st.caption(
        "The following analysis uses actual dataset records. "
        "It displays descriptive percentages and does not represent "
        "individual predictions or causal relationships."
    )

    # Identify categorical features
    categorical_features = [
        col for col in df.columns
        if col not in ["Age", target_col, "Age_Group"]
        and df[col].nunique() <= 10
    ]

    if categorical_features:

        selected_feature = st.selectbox(
            "Select a health factor to analyse",
            categorical_features
        )

        # Create analysis dataframe
        analysis_df = df.copy()

        analysis_df["_positive"] = (
            target_values == 1
        ).astype(int)

        # Group records by selected feature
        grouped = (
            analysis_df.groupby(
                selected_feature,
                dropna=False
            )["_positive"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # Calculate positive percentage
        grouped["Positive Percentage"] = (
            grouped["mean"] * 100
        )

        # Interactive bar chart
        fig2 = px.bar(
            grouped,
            x=selected_feature,
            y="Positive Percentage",
            text_auto=".2f",
            title=f"Percentage of Positive Cases by {selected_feature}",
            labels={
                selected_feature: selected_feature,
                "Positive Percentage": "Positive Cases (%)"
            },
            hover_data={
                "count": True,
                "Positive Percentage": ":.2f"
            }
        )

        fig2.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Positive Cases (%)",
            xaxis_title=selected_feature,
            title_x=0.5
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # Detailed results
        st.subheader("Detailed Results")

        results_df = grouped[
            [selected_feature, "count", "Positive Percentage"]
        ].copy()

        results_df = results_df.rename(
            columns={
                "count": "Total Records"
            }
        )

        results_df["Positive Percentage"] = (
            results_df["Positive Percentage"].round(2)
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "No suitable categorical features were found."
        )

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")

except Exception as e:
    st.error(f"An error occurred: {e}")

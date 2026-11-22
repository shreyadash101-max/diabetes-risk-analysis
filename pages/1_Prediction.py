import streamlit as st
import pandas as pd
import joblib

# Load all four trained models
models = {
    "Random Forest": joblib.load("models/diabetes_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl"),
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Gradient Boosting": joblib.load("models/gradient_boosting_model_fixed.pkl")
}

# Load Random Forest metadata
metadata = joblib.load("models/model_metadata.pkl")
threshold = metadata["threshold"]

# Cross-validation F1 scores
model_scores = {
    "Random Forest": 0.952410,
    "SVM": 0.952291,
    "Gradient Boosting": 0.935458,
    "Logistic Regression": 0.924193
}

st.title("Diabetes Risk Prediction")

st.write(
    "Enter the following information to compare predictions "
    "from four machine learning models."
)

st.info(
    "Educational prototype only. These predictions are not "
    "medical diagnoses."
)

# Input form
with st.form("diabetes_prediction_form"):

    st.subheader("Basic Information")

    age = st.number_input(
        "Age", min_value=1, max_value=120, value=30
    )

    gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("Symptoms and Health Factors")

    polyuria = st.selectbox("Polyuria", ["No", "Yes"])
    polydipsia = st.selectbox("Polydipsia", ["No", "Yes"])
    sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
    weakness = st.selectbox("Weakness", ["No", "Yes"])
    polyphagia = st.selectbox("Polyphagia", ["No", "Yes"])
    genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
    visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
    itching = st.selectbox("Itching", ["No", "Yes"])
    irritability = st.selectbox("Irritability", ["No", "Yes"])
    delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"])
    partial_paresis = st.selectbox("Partial Paresis", ["No", "Yes"])
    muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"])
    alopecia = st.selectbox("Alopecia", ["No", "Yes"])
    obesity = st.selectbox("Obesity", ["No", "Yes"])

    submitted = st.form_submit_button("Compare Predictions")

if submitted:

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Polyuria": polyuria,
        "Polydipsia": polydipsia,
        "sudden weight loss": sudden_weight_loss,
        "weakness": weakness,
        "Polyphagia": polyphagia,
        "Genital thrush": genital_thrush,
        "visual blurring": visual_blurring,
        "Itching": itching,
        "Irritability": irritability,
        "delayed healing": delayed_healing,
        "partial paresis": partial_paresis,
        "muscle stiffness": muscle_stiffness,
        "Alopecia": alopecia,
        "Obesity": obesity
    }])

    results = []

    st.subheader("Model Prediction Comparison")

    for name, model in models.items():

        probability = model.predict_proba(input_data)[0][1]

        # Use the existing threshold for Random Forest.
        # Other classifiers use their standard 0.5 threshold.
        model_threshold = threshold if name == "Random Forest" else 0.5

        prediction = probability >= model_threshold

        results.append({
            "Model": name,
            "Prediction": "Positive" if prediction else "Negative",
            "Positive-class probability": f"{probability * 100:.2f}%",
            "CV F1 Score": f"{model_scores[name] * 100:.2f}%"
        })

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Individual Model Results")

    for result in results:
        with st.expander(result["Model"]):
            st.write("Prediction:", result["Prediction"])
            st.write(
                "Positive-class probability:",
                result["Positive-class probability"]
            )
            st.write("CV F1 Score:", result["CV F1 Score"])

    st.success(
        "Best-performing model: Random Forest "
        "(Mean CV F1 Score: 95.24%)"
    )

    st.caption(
        "Random Forest is highlighted based on its cross-validation "
        "performance, not because it is necessarily the most accurate "
        "for this individual prediction."
    )

    st.caption(
        "Probabilities are model outputs and are not clinically "
        "validated individual diabetes-risk estimates."
    )

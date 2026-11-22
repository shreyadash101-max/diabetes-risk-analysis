# Explainable AI-Based Diabetes Risk Prediction and Risk Factor Analysis System

[![Live Application](https://img.shields.io/badge/Live%20App-Streamlit-red?logo=streamlit)](https://diabetes-risk-analysis-kiit.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)

## Live Application

**Try the application here:**  
https://diabetes-risk-analysis-kiit.streamlit.app/

## 1. Project Overview

The Explainable AI-Based Diabetes Risk Prediction and Risk Factor Analysis System is a machine learning-based web application designed to predict diabetes risk using patient-provided health information and symptoms.

The system combines machine learning prediction models with Explainable Artificial Intelligence (XAI) techniques to provide insights into the factors influencing predictions.

The application uses Streamlit to provide an interactive interface through which users can explore diabetes risk predictions, analyze important risk factors, understand model behaviour and compare machine learning algorithms.

This project is developed as an educational and academic machine learning application.

## 2. Project Objectives

- Develop a machine learning-based diabetes risk prediction system.
- Perform data preprocessing and exploratory data analysis.
- Train and evaluate multiple machine learning algorithms.
- Identify important features associated with diabetes predictions.
- Implement SHAP-based explainability.
- Compare model performance using multiple evaluation metrics.
- Develop an interactive Streamlit application.
- Deploy the application using Streamlit Community Cloud.

## 3. Key Features

### 3.1 Diabetes Risk Prediction
- Accepts age, gender and symptom-related inputs.
- Uses a trained machine learning pipeline to generate predictions.
- Displays the predicted class and estimated probability.

### 3.2 Risk Factor Analysis
- Examines the importance of different input features.
- Provides visual analysis of factors associated with model predictions.
- Includes interactive visualizations.

### 3.3 SHAP Explainability
- Uses SHAP (SHapley Additive exPlanations) to interpret model predictions.
- Visualizes feature contributions.
- Helps explain how individual features influence model output.

### 3.4 Model Evaluation
- Presents model performance using multiple evaluation metrics.
- Includes five-fold cross-validation results.
- Provides performance visualizations and confusion matrix analysis.

### 3.5 Model Comparison
Compares nine machine learning models/configurations across three train-test split ratios.

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Additional ML Libraries | XGBoost, LightGBM, CatBoost |
| Explainable AI | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Excel Processing | OpenPyXL |
| Version Control | Git, GitHub |
| Cloud Deployment | Streamlit Community Cloud |

## 5. Dataset Description

The project uses the Early Stage Diabetes Risk Prediction dataset.

- Original dataset: 520 records.
- Original attributes: 17 columns.
- Duplicate records were removed during preprocessing.
- Cleaned dataset: 251 records.
- Positive cases: 173.
- Negative cases: 78.

The machine learning model uses 16 input features, including age, gender and symptom-related indicators.

The target variable represents the diabetes classification:

- Positive
- Negative

The derived Age_Group column is used for analysis and is not included among the model's 16 input features.

## 6. Machine Learning Models

Nine models/configurations were evaluated:

1. Random Forest
2. Gradient Boosting
3. Logistic Regression
4. Support Vector Machine (SVM)
5. XGBoost
6. LightGBM
7. CatBoost
8. Voting Classifier
9. Stacking Classifier

The models were evaluated using three train-test split ratios:

- 80:20
- 70:30
- 90:10

This resulted in 27 model and split experiments.

## 7. Model Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Precision
- F1 Score
- Area Under the Precision-Recall Curve (AUPR)
- Average Precision (AP)
- Balanced Accuracy
- ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)

Five-fold stratified cross-validation was also performed for Random Forest, SVM, Gradient Boosting and Logistic Regression.

## 8. Explainable AI Using SHAP

SHAP (SHapley Additive exPlanations) is used to understand how individual input features contribute to machine learning predictions.

The project includes SHAP visualizations to examine feature importance and feature contributions.

Polyuria and Polydipsia were among the most influential features in the model's feature-importance analysis.

SHAP improves the interpretability of the model by providing insights into its decision-making process.

## 9. Project Structure

```text
diabetes-risk-analysis/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── diabetes_data_upload.csv
│
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_model_fixed.pkl
│   ├── gradient_boosting_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl
│   ├── model_metadata.pkl
│   ├── feature_importance.csv
│   └── final_metrics.csv
│
├── pages/
│   ├── 1_Prediction.py
│   ├── 2_Risk_Factor_Analysis.py
│   ├── 3_SHAP_Explainability.py
│   ├── 4_Model_Evaluation.py
│   └── 5_Model_Comparison.py
│
├── model_comparison/
│   └── results/
│       └── Diabetes_9_Model_Comparison_Results.xlsx
│
├── shap_exports/
│   ├── shap_feature_importance.png
│   ├── shap_grouped_contributions.png
│   ├── shap_grouped_importance.png
│   └── shap_summary.png
│
└── retrain_gradient_boosting.py
```

## 10. Installation and Execution

To run the application locally:

### Step 1: Clone the repository

```bash
git clone https://github.com/shreyadash101-max/diabetes-risk-analysis.git
```

### Step 2: Navigate to the project directory

```bash
cd diabetes-risk-analysis
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

## 11. Cloud Deployment

The application is deployed using Streamlit Community Cloud.

Deployment workflow:

1. Source code is maintained in GitHub.
2. The GitHub repository is connected to Streamlit Community Cloud.
3. The main application file (`app.py`) is selected.
4. Dependencies are installed using `requirements.txt`.
5. The application is deployed and made accessible through a public URL.

**Live Application:**  
https://diabetes-risk-analysis-kiit.streamlit.app/

## 12. Future Enhancements

- Validate the system using larger and more diverse datasets.
- Explore additional machine learning and ensemble techniques.
- Improve model generalization through further validation.
- Enhance individual prediction explanations.
- Improve application usability and visualization.
- Evaluate model performance on independent datasets.

## 13. Disclaimer

This application is developed for educational and academic purposes only. It is not a clinically validated diagnostic tool and must not be used as a substitute for professional medical advice, diagnosis or treatment.

## 14. Author

**Shreya Dash**  
B.Tech – Computer Science and Engineering  
Kalinga Institute of Industrial Technology (KIIT)

GitHub: https://github.com/shreyadash101-max

---

**Project:** Explainable AI-Based Diabetes Risk Prediction and Risk Factor Analysis System  
**Deployment:** Streamlit Community Cloud
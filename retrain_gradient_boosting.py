
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset
df = pd.read_csv("data/diabetes_data_upload.csv")

# Select the original 16 input features
numeric_features = ["Age"]

categorical_features = [
    "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
    "weakness", "Polyphagia", "Genital thrush", "visual blurring",
    "Itching", "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity"
]

X = df[numeric_features + categorical_features]
y = df["class"]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# Gradient Boosting model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ))
])

# Same 80:20 split and random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Save
with open("models/gradient_boosting_model_fixed.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)

print("Gradient Boosting retrained and saved successfully!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Model type:", type(model))

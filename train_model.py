import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("loan_data.csv")

print(df.head())
print(df.info())
print(df.shape)

# ==============================
# ตรวจสอบ Missing Value
# ==============================
print("\nMissing Value")
print(df.isnull().sum())

df = df.dropna()

# ==============================
# Feature และ Target
# ==============================
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ==============================
# แยกข้อมูลตัวเลขและข้อความ
# ==============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumeric Features")
print(numeric_features)

print("\nCategorical Features")
print(categorical_features)

# ==============================
# Data Preprocessing
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ==============================
# Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# สร้างโมเดล SVM
# ==============================
svm_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            kernel="rbf",
            C=1,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ]
)

# ==============================
# Train Model
# ==============================
print("\nTraining Model...")
svm_model.fit(X_train, y_train)

# ==============================
# Evaluation
# ==============================
y_pred = svm_model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==============================
# ทดลองทำนาย
# **ตรวจสอบค่าข้อความให้ตรงกับ Dataset**
# ==============================
sample = pd.DataFrame({

    "person_age": [25],
    "person_gender": ["male"],
    "person_education": ["Bachelor"],
    "person_income": [55000],
    "person_emp_exp": [4],
    "person_home_ownership": ["RENT"],
    "loan_amnt": [12000],
    "loan_intent": ["EDUCATION"],
    "loan_int_rate": [12.5],
    "loan_percent_income": [0.25],
    "cb_person_cred_hist_length": [3],
    "credit_score": [710],
    "previous_loan_defaults_on_file": ["No"]

})

prediction = svm_model.predict(sample)
probability = svm_model.predict_proba(sample)

print("\nPrediction")
print(prediction)

print("\nProbability")
print(probability)

# ==============================
# Save Model
# ==============================
with open("loan_svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

print("\nModel Saved Successfully")
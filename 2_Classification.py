import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Title
st.title("🧠 Classification Model Comparison Dashboard")

# Create dummy classification data (Employee Attrition)
np.random.seed(0)
df = pd.DataFrame({
    "Age": np.random.randint(20, 60, 100),
    "MonthlyIncome": np.random.randint(3000, 15000, 100),
    "YearsAtCompany": np.random.randint(0, 20, 100),
    "JobSatisfaction": np.random.randint(1, 5, 100),
    "OverTime": np.random.choice([0, 1], 100),  # 0: No, 1: Yes
})

# Generate target based on a formula (Attrition)
df["Attrition"] = ((df["OverTime"] == 1) & (df["JobSatisfaction"] < 3) & (df["YearsAtCompany"] < 5)).astype(int)

if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Split data
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize for specific models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = []

st.subheader("📊 Model Performance")

for name, model in models.items():
    if name in ["Logistic Regression", "SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except:
        roc = float("nan")

    st.markdown(f"### 🔹 {name}")
    st.write(f"- Accuracy: {acc:.2f}")
    st.write(f"- Precision: {prec:.2f}")
    st.write(f"- Recall: {rec:.2f}")
    st.write(f"- F1 Score: {f1:.2f}")
    st.write(f"- ROC AUC: {roc:.2f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc
    })

# Results summary
st.subheader("📋 Model Comparison Summary")
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
st.dataframe(results_df)

best_model = results_df.iloc[0]
st.success(f"✅ Best Model: {best_model['Model']} with F1 Score = {best_model['F1 Score']:.2f}")

st.markdown("---")
st.info("This app demonstrates supervised classification using multiple models and evaluates them using key classification metrics.")

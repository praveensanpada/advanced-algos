import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title
st.title("🧠 Classification Model Comparison Dashboard")

# Dummy classification dataset
np.random.seed(42)
df = pd.DataFrame({
    'Age': np.random.randint(18, 60, 50),
    'Salary': np.random.randint(30000, 120000, 50),
    'Experience': np.random.randint(1, 15, 50),
    'Education_Level': np.random.choice([0, 1, 2], 50),  # 0: High School, 1: Graduate, 2: Postgraduate
    'Married': np.random.choice([0, 1], 50),  # 0: No, 1: Yes
})

# Ensure roughly equal 0s and 1s
condition = np.random.choice([0, 1], size=50, p=[0.5, 0.5])
df['Target'] = condition

# Show raw data
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Features and Target
X = df.drop("Target", axis=1)
y = df["Target"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data for models like SVM, KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selector
model_name = st.selectbox("Select Classification Model", [
    "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"])

# Initialize model
if model_name == "Logistic Regression":
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

elif model_name == "Random Forest":
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

elif model_name == "SVM":
    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

elif model_name == "KNN":
    model = KNeighborsClassifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.subheader("📊 Evaluation Metrics")
st.write(f"- Accuracy: {accuracy:.2f}")
st.write(f"- Precision: {precision:.2f}")
st.write(f"- Recall: {recall:.2f}")
st.write(f"- F1 Score: {f1:.2f}")

try:
    roc = roc_auc_score(y_test, y_prob)
    st.write(f"- ROC AUC Score: {roc:.2f}")
except ValueError:
    st.warning("ROC AUC score is not defined because y_test contains only one class.")

# Classification Report
st.subheader("📝 Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")
fig, ax = plt.subplots()
display = ConfusionMatrixDisplay.from_estimator(model, X_test_scaled if model_name in ["SVM", "KNN", "Logistic Regression"] else X_test, y_test, ax=ax)
st.pyplot(fig)

st.info("This app compares multiple classification models using accuracy, precision, recall, F1-score, and ROC-AUC.")
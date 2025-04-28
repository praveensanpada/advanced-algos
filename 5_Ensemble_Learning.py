import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Title
st.title("🚀 Advanced Machine Learning: Feature Engineering + Ensemble Model Comparison")

# Generate dummy classification data
X, y = make_classification(n_samples=200, n_features=10, n_informative=6, n_redundant=2, random_state=42)
columns = [f"Feature_{i+1}" for i in range(10)]
df = pd.DataFrame(X, columns=columns)
df['Target'] = y

if st.checkbox("Show Raw Dataset"):
    st.dataframe(df.head())

# Feature Selection
st.subheader("🎯 Feature Selection using SelectKBest")
k = st.slider("Select number of top features", 2, 10, 5)
X_selected = SelectKBest(score_func=f_classif, k=k).fit_transform(X, y)

# Dimensionality Reduction for Visualization
st.subheader("🧬 Dimensionality Reduction")
dim_reduction = st.selectbox("Choose Method", ["PCA", "t-SNE"])

X_scaled = StandardScaler().fit_transform(X_selected)
if dim_reduction == "PCA":
    X_reduced = PCA(n_components=2).fit_transform(X_scaled)
else:
    X_reduced = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(X_scaled)

reduced_df = pd.DataFrame(X_reduced, columns=['Dim1', 'Dim2'])
reduced_df['Target'] = y
fig, ax = plt.subplots()
sns.scatterplot(data=reduced_df, x='Dim1', y='Dim2', hue='Target', palette='coolwarm', ax=ax)
st.pyplot(fig)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier()
}

# Model evaluation
st.subheader("📊 Ensemble Model Evaluation")
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

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

# Comparison Table
st.subheader("📋 Ensemble Model Comparison Summary")
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
st.dataframe(results_df)

# Best model
best_model = results_df.iloc[0]
st.success(f"✅ Best Performing Model: {best_model['Model']} with F1 Score = {best_model['F1 Score']:.2f}")

st.markdown("---")
st.info("This app demonstrates feature selection, dimensionality reduction, and ensemble learning to evaluate model performance on classification tasks.")
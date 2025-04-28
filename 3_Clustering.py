import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Title
st.title("🔍 Customer Segmentation with Clustering Algorithms")

# Generate realistic customer data
df = pd.DataFrame({
    'Annual_Income_k': np.random.randint(20, 150, 100),
    'Spending_Score': np.random.randint(1, 100, 100),
    'Age': np.random.randint(18, 65, 100),
    'Loyalty_Years': np.random.randint(1, 15, 100),
    'Num_Transactions': np.random.randint(10, 200, 100)
})

if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Define clustering models
models = {
    'K-Means (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
    'Hierarchical (k=3)': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=1.5, min_samples=5)
}

results = []

st.subheader("📊 Clustering Model Performance")

for name, model in models.items():
    if name == 'DBSCAN':
        labels = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        labels = model.fit_predict(X_scaled)
        n_clusters = len(np.unique(labels))

    # Evaluate using silhouette score (only if >1 cluster)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
    else:
        silhouette = float("nan")

    results.append({
        "Model": name,
        "Clusters": n_clusters,
        "Silhouette Score": silhouette
    })

    # Visualization
    st.markdown(f"### 🔹 {name}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Annual_Income_k'], y=df['Spending_Score'], hue=labels, palette='viridis', ax=ax)
    plt.xlabel("Annual Income (k)")
    plt.ylabel("Spending Score")
    st.pyplot(fig)

# Display comparison table
st.subheader("📋 Clustering Model Comparison Summary")
results_df = pd.DataFrame(results).sort_values(by="Silhouette Score", ascending=False)
st.dataframe(results_df)

# Show best model
best_model = results_df.iloc[0]
st.success(f"✅ Best Clustering Model: {best_model['Model']} with Silhouette Score = {best_model['Silhouette Score']:.2f}")

# Dendrogram for Hierarchical Clustering
st.subheader("🧬 Dendrogram (Hierarchical Clustering)")
linkage_matrix = linkage(X_scaled, method='ward')
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
st.pyplot(fig)

st.markdown("---")
st.info("This app applies multiple clustering algorithms on customer data, visualizes the results, and uses silhouette scores to compare and identify the best model.")

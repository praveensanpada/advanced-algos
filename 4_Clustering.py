import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Title
st.title("🔍 Customer Segmentation using Clustering")

# Generate dummy customer data
np.random.seed(42)
df = pd.DataFrame({
    'Annual_Income_k': np.random.randint(20, 150, 50),
    'Spending_Score': np.random.randint(1, 100, 50),
    'Age': np.random.randint(18, 65, 50),
    'Loyalty_Years': np.random.randint(1, 10, 50),
    'Num_Transactions': np.random.randint(10, 200, 50)
})

if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# Select clustering method
clustering_method = st.selectbox("Choose Clustering Method", ["K-Means", "Hierarchical Clustering", "DBSCAN"])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply clustering
if clustering_method == "K-Means":
    k = st.slider("Select number of clusters (K)", 2, 10, 3)
    # model = KMeans(n_clusters=k, random_state=42)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    df['Cluster'] = labels

elif clustering_method == "Hierarchical Clustering":
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
    linkage_matrix = linkage(X_scaled, method='ward')
    st.subheader("Dendrogram")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    st.pyplot(fig)

    # Use fixed number of clusters for visualization
    h_model = AgglomerativeClustering(n_clusters=3)
    labels = h_model.fit_predict(X_scaled)
    df['Cluster'] = labels

elif clustering_method == "DBSCAN":
    eps_val = st.slider("Select epsilon (eps)", 0.1, 5.0, 1.5)
    min_samples_val = st.slider("Select min_samples", 2, 10, 5)
    model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = model.fit_predict(X_scaled)
    df['Cluster'] = labels

# Plot clusters
if 'Cluster' in df.columns:
    st.subheader("Cluster Visualization (Income vs Spending Score)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='Annual_Income_k', y='Spending_Score', hue='Cluster', palette='viridis', ax=ax2)
    st.pyplot(fig2)

# Show clustered data
if 'Cluster' in df.columns:
    st.subheader("Clustered Customer Data")
    st.dataframe(df)

st.markdown("---")
st.info("This project demonstrates customer segmentation using K-Means, DBSCAN, and Hierarchical Clustering. You can use it to identify marketing groups, loyal customers, or behavior-based segments.")

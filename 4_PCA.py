import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Create synthetic data: 100 rows and 5 columns
np.random.seed(42)
data = np.random.rand(100, 5) * 100  # Random data scaled between 0 and 100
columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
df = pd.DataFrame(data, columns=columns)

# 2. Standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Applying PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
df_pca = pca.fit_transform(df_scaled)

# 4. Create a DataFrame with PCA results
df_pca_df = pd.DataFrame(df_pca, columns=['Principal Component 1', 'Principal Component 2'])

# 5. Streamlit UI
st.title('PCA Example - Streamlit')
st.subheader('Original Data')
st.write(df.head())

# 6. Display PCA explained variance ratio
st.subheader('Explained Variance by Principal Components')
explained_variance = pca.explained_variance_ratio_
st.write(f'Principal Component 1: {explained_variance[0]:.2f}')
st.write(f'Principal Component 2: {explained_variance[1]:.2f}')

# 7. Plotting the original data and PCA projection
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plotting original data (just to show that it's high-dimensional)
ax[0].scatter(df['Feature1'], df['Feature2'], color='blue', alpha=0.6)
ax[0].set_title('Original Data (Feature1 vs Feature2)')
ax[0].set_xlabel('Feature1')
ax[0].set_ylabel('Feature2')

# Plotting PCA result
ax[1].scatter(df_pca_df['Principal Component 1'], df_pca_df['Principal Component 2'], color='red', alpha=0.6)
ax[1].set_title('PCA Reduced Data')
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')

st.pyplot(fig)

# 8. Additional Analysis
st.subheader('PCA Results')
st.write(df_pca_df.head())

# 9. Displaying errors if any
try:
    # Simulate a simple error scenario for demonstration
    if df.isnull().values.any():
        raise ValueError("Data contains missing values!")
except Exception as e:
    st.error(f"Error: {e}")

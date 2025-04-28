import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Create synthetic data: 100 rows and 5 features
np.random.seed(42)
data = np.random.rand(100, 5) * 100  # Random data scaled between 0 and 100
columns = ['Income', 'Spending_Score', 'Age', 'Family_Size', 'Education_Level']
df = pd.DataFrame(data, columns=columns)

# 2. Standardizing the data (important for t-SNE)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

# 4. Create a DataFrame with t-SNE results
df_tsne_df = pd.DataFrame(df_tsne, columns=['Principal Component 1', 'Principal Component 2'])

# 5. Streamlit UI
st.title('t-SNE Example - Streamlit')
st.subheader('Original Data')
st.write(df.head())

# 6. Plotting the t-SNE results
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot of t-SNE results
ax.scatter(df_tsne_df['Principal Component 1'], df_tsne_df['Principal Component 2'], color='blue', alpha=0.6)
ax.set_title('t-SNE Reduced Data')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

st.pyplot(fig)

# 7. Displaying the explained variance (not directly available in t-SNE)
st.subheader('t-SNE Results')
st.write(df_tsne_df.head())

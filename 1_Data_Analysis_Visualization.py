import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Retail Sales Dashboard - Seasonal Trend Analysis")

# Generate Dummy Retail Sales Data
np.random.seed(42)
data = {
    "Date": pd.date_range(start="2023-01-01", periods=50, freq='W'),
    "Store": np.random.choice(["Store A", "Store B", "Store C"], size=50),
    "Product": np.random.choice(["Product X", "Product Y", "Product Z"], size=50),
    "Units_Sold": np.random.randint(10, 100, size=50),
    "Revenue": np.random.uniform(500, 5000, size=50).round(2)
}
df = pd.DataFrame(data)

# Convert Date to Month for grouping
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year

# Sidebar Filters
st.sidebar.header("Filter Data")
selected_store = st.sidebar.multiselect("Select Store:", options=df['Store'].unique(), default=df['Store'].unique())
selected_product = st.sidebar.multiselect("Select Product:", options=df['Product'].unique(), default=df['Product'].unique())

# Filtered Data
df_filtered = df[(df['Store'].isin(selected_store)) & (df['Product'].isin(selected_product))]

# Show Raw Data if needed
if st.checkbox("Show Raw Data"):
    st.dataframe(df_filtered)

# Summary Stats
st.subheader("📌 Summary Statistics")
st.write(df_filtered.describe())

# Monthly Sales Trend
st.subheader("📈 Monthly Sales Trend")
monthly_sales = df_filtered.groupby(['Month', 'Product'])['Units_Sold'].sum().reset_index()
monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
monthly_sales = monthly_sales.sort_values('Month')

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=monthly_sales, x='Month', y='Units_Sold', hue='Product', marker='o', ax=ax)
plt.xticks(rotation=45)
plt.title('Units Sold per Product per Month')
st.pyplot(fig)

# Revenue Distribution
st.subheader("💰 Revenue Distribution by Store")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df_filtered, x='Store', y='Revenue', ax=ax2)
plt.title('Revenue Distribution per Store')
st.pyplot(fig2)

# Forecasting Placeholder
st.subheader("🔮 Future Sales Forecast (Placeholder)")
st.info("In a real project, you'd use models like Prophet or ARIMA to forecast future trends. Here, we demonstrate trend analysis visually.")

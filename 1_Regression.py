import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

# Title
st.title("🏡 House Price Prediction: Regression Model Comparison")

# Generate Realistic House Price Data
np.random.seed(0)
data = {
    "Size_sqft": np.random.randint(500, 3000, size=100),
    "Bedrooms": np.random.randint(1, 5, size=100),
    "Age_years": np.random.randint(0, 30, size=100),
    "Bathrooms": np.random.randint(1, 3, size=100),
    "Location_Score": np.random.randint(1, 10, size=100),
}
df = pd.DataFrame(data)

# Create a target variable simulating house prices
noise = np.random.normal(0, 25000, 100)
df['Price'] = (
    df['Size_sqft'] * 150 +
    df['Bedrooms'] * 10000 +
    df['Bathrooms'] * 15000 +
    df['Location_Score'] * 20000 -
    df['Age_years'] * 1000 +
    noise
)

if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# Split dataset
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper for polynomial regression
def make_polynomial_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Models dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression (Degree 2)': lambda: make_polynomial_model(2),
    'Polynomial Regression (Degree 3)': lambda: make_polynomial_model(3),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Container for results
results = []

st.subheader("📈 Model Performance")

for name, model in models.items():
    if "Polynomial" in name:
        reg = model()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
    elif name in ['Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"### 🔹 {name}")
    st.write(f"- MSE: {mse:.2f}")
    st.write(f"- MAE: {mae:.2f}")
    st.write(f"- RMSE: {rmse:.2f}")
    st.write(f"- R² Score: {r2:.2f}")

    results.append({"Model": name, "MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2})

# Summary Comparison Table
st.subheader("📊 Model Comparison Summary")
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
st.dataframe(results_df)

best_model = results_df.iloc[0]
st.success(f"✅ Best Model: {best_model['Model']} with R² = {best_model['R2']:.2f}")

st.markdown("---")
st.info("This app uses real-world-like housing data to train and compare various regression models, helping to identify the best one based on performance metrics.")

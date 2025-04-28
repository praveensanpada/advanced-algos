import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Title
st.title("🏠 House Price Prediction using Regression")

# Generate dummy dataset
np.random.seed(0)
data = {
    "Size_sqft": np.random.randint(500, 3000, size=50),
    "Bedrooms": np.random.randint(1, 5, size=50),
    "Age_years": np.random.randint(0, 30, size=50),
    "Bathrooms": np.random.randint(1, 3, size=50),
    "Location_Score": np.random.randint(1, 10, size=50),
}
df = pd.DataFrame(data)
df["Price"] = 50000 + df["Size_sqft"] * 3 + df["Bedrooms"] * 10000 + df["Location_Score"] * 1500 - df["Age_years"] * 1000 + np.random.normal(0, 10000, size=50)

# Show raw data
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Feature and Target Selection
X = df.drop("Price", axis=1)
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model type selector
model_type = st.selectbox("Choose Regression Model", ["Linear Regression", "Polynomial Regression"])

if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_type == "Polynomial Regression":
    degree = st.slider("Select Degree", min_value=2, max_value=5, value=2)
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Evaluation Metrics")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"- R² Score: {r2:.2f}")

# Bias vs Variance: Show train vs test error using cross-validation
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')
st.subheader("📈 Cross-Validation R² Scores")
st.write(cv_scores)
st.write(f"Average R²: {np.mean(cv_scores):.2f}")

# Plot true vs predicted prices
st.subheader("🏘️ True vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.set_title('Actual vs Predicted House Prices')
st.pyplot(fig)

st.info("This project demonstrates ML fundamentals: model selection, bias-variance, train-test split, and metrics. You can extend it with more algorithms or regularization.")

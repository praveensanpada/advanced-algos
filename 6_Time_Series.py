import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Title
st.title("📈 Time Series Forecasting Comparison: ARIMA, Prophet, LSTM")

# Generate dummy time series data (e.g., daily sales)
dates = pd.date_range(start='2023-01-01', periods=200)
sales = (np.sin(np.linspace(0, 20, 200)) * 100 + np.random.normal(0, 10, 200) + 300).round(2)
df = pd.DataFrame({'ds': dates, 'y': sales})

if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Plot time series
st.subheader("📊 Original Time Series")
fig, ax = plt.subplots()
ax.plot(df['ds'], df['y'])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Daily Sales")
st.pyplot(fig)

# Forecast horizon
future_periods = st.slider("Select forecast horizon (days)", 10, 60, 30)

# --- ARIMA Forecast ---
st.subheader("🔹 ARIMA Forecast")
model_arima = ARIMA(df['y'], order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=future_periods)

fig1, ax1 = plt.subplots()
ax1.plot(df['ds'], df['y'], label='Actual')
ax1.plot(pd.date_range(df['ds'].iloc[-1] + timedelta(days=1), periods=future_periods), forecast_arima, label='ARIMA Forecast')
plt.legend()
st.pyplot(fig1)

# --- Prophet Forecast ---
st.subheader("🔹 Prophet Forecast")
model_prophet = Prophet()
model_prophet.fit(df)
future_df = model_prophet.make_future_dataframe(periods=future_periods)
forecast_prophet = model_prophet.predict(future_df)

fig2 = model_prophet.plot(forecast_prophet)
st.pyplot(fig2)

# --- LSTM Forecast ---
st.subheader("🔹 LSTM Forecast")
data = df[['y']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 10
X_lstm, y_lstm = create_sequences(data_scaled, TIME_STEPS)
X_train, y_train = X_lstm[:-future_periods], y_lstm[:-future_periods]

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(TIME_STEPS, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=20, verbose=0)

last_sequence = data_scaled[-TIME_STEPS:]
predictions = []
current_seq = last_sequence.reshape(1, TIME_STEPS, 1)
for _ in range(future_periods):
    next_val = model_lstm.predict(current_seq, verbose=0)[0, 0]
    predictions.append(next_val)
    current_seq = np.append(current_seq[:, 1:, :], [[[next_val]]], axis=1)

predicted_lstm = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

fig3, ax3 = plt.subplots()
ax3.plot(df['ds'], df['y'], label='Actual')
ax3.plot(pd.date_range(df['ds'].iloc[-1] + timedelta(days=1), periods=future_periods), predicted_lstm, label='LSTM Forecast')
plt.legend()
st.pyplot(fig3)

st.markdown("---")
st.success("✅ Comparison Complete! You can now evaluate which forecasting method works best based on trend consistency, shape, and forecast range.")

st.info("This app compares ARIMA, Prophet, and LSTM models for time series forecasting using simulated sales data. Tune horizon and explore visual trends.")

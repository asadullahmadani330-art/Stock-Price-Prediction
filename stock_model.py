import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




print("Downloading historical stock data...")


df = yf.download("TSLA", start="2015-01-01", end="2024-12-31")

print("Stock data downloaded successfully.\n")




print("Preprocessing data...")

# Select relevant features
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]

# Create target variable (Next Day Closing Price)
df['Target'] = df['Close'].shift(-1)

# Remove last row with NaN target
df.dropna(inplace=True)

print("Data preprocessing completed.\n")


# ----------------------------------------------
# Feature Selection
# ----------------------------------------------

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Target']


# ----------------------------------------------
# Time-Based Train-Test Split (80% Train, 20% Test)
# ----------------------------------------------

print("Splitting dataset...")

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Dataset split completed.\n")



print("Training Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed.\n")




print("Making predictions...")

y_pred = model.predict(X_test)




print("\nModel Evaluation Results")
print("=" * 45)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")



results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv")


# ----------------------------------------------
# Visualization
# ----------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.title("Tesla (TSLA) Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.tight_layout()
plt.show()

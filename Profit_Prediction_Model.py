import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')

# --- Preprocessing ---
# Drop rows where essential columns are missing
df = df.dropna(subset=['Quantity', 'UnitPrice', 'Profit'])

# Optional: Create extra features
df['Total_Sale'] = df['Quantity'] * df['UnitPrice']

# Select features and target
features = ['Quantity', 'UnitPrice', 'Total_Sale']
target = 'Profit'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📈 MSE: {mse:.2f}")
print(f"📊 R2 Score: {r2:.2f}")

# Save model
joblib.dump(model, "profit_prediction_model.pkl")

print("✅ Profit prediction model created and saved.")

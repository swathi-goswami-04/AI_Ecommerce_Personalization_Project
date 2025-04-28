import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')

# --- Define Fraud Label (Rule-based) ---
# Mark as fraud if:
# - High discount (> 0.5)
# - High quantity (> 5)
# - Low or negative profit
df['Is_Fraud'] = df.apply(lambda row: 1 if (row['Discount'] > 0.5 and row['Quantity'] > 5 and row['Profit'] < 0) else 0, axis=1)

# --- Feature Engineering ---
features_df = df[[
    'Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost',
    'Device_Type', 'Customer_Login_type', 'Is_Fraud'
]].copy()

# One-hot encode categorical features
features_df = pd.get_dummies(features_df, columns=['Device_Type', 'Customer_Login_type'])

# Split features and label
X = features_df.drop('Is_Fraud', axis=1)
y = features_df['Is_Fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluation
y_pred = clf.predict(X_test_scaled)
print("🚨 Fraud Detection Report:\n", classification_report(y_test, y_pred))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(clf, "fraud_rf_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")
joblib.dump(X.columns.tolist(), "fraud_features.pkl")

# Save predictions
df['Predicted_Fraud'] = clf.predict(scaler.transform(X))
df.to_csv("fraud_predictions.csv", index=False)

print("✅ Fraud detection complete. Artifacts saved.")

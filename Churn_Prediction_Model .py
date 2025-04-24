import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')

# Reference date
today = datetime(2025, 4, 12)

# --- Feature Engineering ---
customer_df = df.groupby('Customer_Id').agg({
    'Sales': 'sum',
    'Order_Date': lambda x: (today - x.max()).days,
    'Profit': 'mean',
    'Discount': 'mean',
    'Quantity': 'mean',
    'Order_Id': 'nunique'
}).reset_index()

customer_df.columns = ['Customer_Id', 'Total_Spent', 'Recency', 'Avg_Profit', 'Avg_Discount', 'Avg_Quantity', 'Order_Count']

# Most used device and login type
extra_features = df.groupby('Customer_Id').agg({
    'Device_Type': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'Customer_Login_type': lambda x: x.mode()[0] if not x.mode().empty else np.nan
}).reset_index()

customer_df = customer_df.merge(extra_features, on='Customer_Id', how='left')

# One-hot encode
customer_df = pd.get_dummies(customer_df, columns=['Device_Type', 'Customer_Login_type'])

# --- Define Churn Logic ---
# Mark customers as churned if Recency > 90 days
customer_df['Churn'] = customer_df['Recency'].apply(lambda x: 1 if x > 90 else 0)

# Prepare data
X = customer_df.drop(['Customer_Id', 'Churn'], axis=1)
y = customer_df['Churn']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = clf.predict(X_test_scaled)
print("🎯 Classification Report:\n", classification_report(y_test, y_pred))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save Artifacts ---
joblib.dump(clf, "churn_rf_model.pkl")
joblib.dump(scaler, "churn_scaler.pkl")

# Add churn prediction to dataframe
customer_df['Predicted_Churn'] = clf.predict(scaler.transform(X))

# Save results
customer_df.to_csv("customer_churn_predictions.csv", index=False)

print("✅ Churn prediction complete. Model, scaler, and results saved.")


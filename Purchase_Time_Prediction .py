import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')

# Drop nulls in key columns
df = df.dropna(subset=['Order_Date', 'Customer_Id', 'Quantity', 'Sales'])


# Extract hour from timestamp
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True)
df['Hour'] = df['Order_Date'].dt.hour

def get_time_bin(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

df['TimeBin'] = df['Hour'].apply(get_time_bin)
df['TotalAmount'] = df['Sales']

features = ['Quantity', 'Sales', 'TotalAmount']
X = df[features]
y = df['TimeBin'].astype('category').cat.codes


# Encode target
y = y.astype('category').cat.codes  # Morning=0, Afternoon=1, Evening=2, Night=3

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(clf, "purchase_time_prediction_model.pkl")

print("🎯 Purchase Time Prediction model saved successfully.")

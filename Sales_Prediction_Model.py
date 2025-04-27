import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

#  Load the dataset
df = pd.read_csv('E_commerce_Dataset.csv')

#  Preprocessing
df['Product'] = pd.Categorical(df['Product']).codes
df['Product_Category'] = pd.Categorical(df['Product_Category']).codes
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
df['Day'] = df['Order_Date'].dt.day
df['Month'] = df['Order_Date'].dt.month
df['Year'] = df['Order_Date'].dt.year

#  Define Features and Target
X = df.drop(['Sales'], axis=1)
y = df['Sales']

# Drop unnecessary columns
X = X.drop(columns=['Order_Date', 'Time'], errors='ignore')
X = X.select_dtypes(include=[np.number])

#  Remove rows with NaNs
Xy = pd.concat([X, y], axis=1).dropna()
X = Xy.drop('Sales', axis=1)
y = Xy['Sales']

#  Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Scale selective features
scaler = StandardScaler()
X_train[['Quantity', 'Discount', 'Shipping_Cost']] = scaler.fit_transform(X_train[['Quantity', 'Discount', 'Shipping_Cost']])
X_test[['Quantity', 'Discount', 'Shipping_Cost']] = scaler.transform(X_test[['Quantity', 'Discount', 'Shipping_Cost']])

#  Train XGBoost Model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

#  Save Model and Scaler
xgb_model.save_model("xgb_model.json")
joblib.dump(scaler, "sales_scaler.pkl")
print("✅ XGBoost model saved as xgb_model.json")
print("✅ Sales scaler saved as sales_scaler.pkl")

#  Evaluate Model
y_pred_xgb = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Regressor MSE: {xgb_mse:.3f}, MAE: {xgb_mae:.3f}, R2: {xgb_r2:.3f}')

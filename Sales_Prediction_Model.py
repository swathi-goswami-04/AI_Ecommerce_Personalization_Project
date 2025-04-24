import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer


# Load the data
df = pd.read_csv('E_commerce_Dataset.csv')

# Preprocess the data
df['Product'] = pd.Categorical(df['Product']).codes
df['Product_Category'] = pd.Categorical(df['Product_Category']).codes
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
df['Day'] = df['Order_Date'].dt.day
df['Month'] = df['Order_Date'].dt.month
df['Year'] = df['Order_Date'].dt.year

# Define the features and target
X = df.drop(['Sales'], axis=1)
y = df['Sales']

# Drop datetime/string columns that cause issues
X = X.drop(columns=['Order_Date', 'Time'], errors='ignore')

# Optional: Ensure only numerical data
X = X.select_dtypes(include=[np.number])

Xy = pd.concat([X, y], axis=1).dropna()
X = Xy.drop('Sales', axis=1)
y = Xy['Sales']



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train[['Quantity', 'Discount', 'Shipping_Cost']] = scaler.fit_transform(X_train[['Quantity', 'Discount', 'Shipping_Cost']])
X_test[['Quantity', 'Discount', 'Shipping_Cost']] = scaler.transform(X_test[['Quantity', 'Discount', 'Shipping_Cost']])

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate the Linear Regression model
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
print(f'Linear Regression MSE: {lr_mse:.3f}, MAE: {lr_mae:.3f}, R2: {lr_r2:.3f}')

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest Regressor model
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
print(f'Random Forest Regressor MSE: {rf_mse:.3f}, MAE: {rf_mae:.3f}, R2: {rf_r2:.3f}')

# XGBoost Regressor
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost Regressor model
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)
print(f'XGBoost Regressor MSE: {xgb_mse:.3f}, MAE: {xgb_mae:.3f}, R2: {xgb_r2:.3f}')

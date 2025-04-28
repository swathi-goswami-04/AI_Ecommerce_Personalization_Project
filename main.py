from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

app = FastAPI(title="🚀 E-commerce ML Backend (Final)", version="2.0")

# --- Load models ---
churn_model = joblib.load("churn_rf_model.pkl")
churn_scaler = joblib.load("churn_scaler.pkl")
churn_features = joblib.load("churn_features.pkl")

segment_model = joblib.load("customer_kmeans_model.pkl")
segment_scaler = joblib.load("customer_scaler.pkl")
segment_features = joblib.load("segment_features.pkl")

fraud_model = joblib.load("fraud_rf_model.pkl")
fraud_scaler = joblib.load("fraud_scaler.pkl")
fraud_features = joblib.load("fraud_features.pkl")

user_product_matrix = joblib.load("user_product_matrix.pkl")
similarity_knn = joblib.load("user_knn_model.pkl")

profit_model = joblib.load("profit_prediction_model.pkl")
purchase_time_model = joblib.load("purchase_time_prediction_model.pkl")

sales_model = XGBRegressor()
sales_model.load_model("xgb_model.json")
sales_scaler = joblib.load("sales_scaler.pkl")
sales_features = joblib.load("sales_features.pkl")

# --- Enums ---
class DeviceType(str, Enum):
    Mobile = "Mobile"
    Web = "Web"

class LoginType(str, Enum):
    FirstSignUp = "First SignUp"
    Guest = "Guest"
    Member = "Member"
    New = "New "  # Note: Space at end!!

# --- Schemas ---
class ChurnInput(BaseModel):
    total_spent: float
    recency: int
    avg_profit: float
    avg_discount: float
    avg_quantity: float
    order_count: int
    device_type: DeviceType
    login_type: LoginType

class SegmentInput(ChurnInput):
    pass

class FraudInput(BaseModel):
    sales: float
    quantity: int
    discount: float
    profit: float
    shipping_cost: float
    device_type: DeviceType
    login_type: LoginType

class RecommendInput(BaseModel):
    customer_id: int

class ProfitInput(BaseModel):
    quantity: float
    discount: float
    sales: float

class PurchaseTimeInput(BaseModel):
    quantity: float
    sales: float

class SalesInput(BaseModel):
    aging: float
    customer_id: int
    product_category: int
    product: int
    quantity: int
    discount: float
    profit: float
    shipping_cost: float
    order_id: int
    day: int
    month: int
    year: int

# --- Helper ---
def align_features(input_data, feature_list):
    df = pd.DataFrame([input_data], columns=feature_list)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_list]
    return df

# --- Endpoints ---
@app.post("/predict/churn")
async def predict_churn(data: ChurnInput):
    features = [
        data.total_spent,
        data.recency,
        data.avg_profit,
        data.avg_discount,
        data.avg_quantity,
        data.order_count,
        1 if data.device_type == "Mobile" else 0,
        1 if data.device_type == "Web" else 0,
        1 if data.login_type == "First SignUp" else 0,
        1 if data.login_type == "Guest" else 0,
        1 if data.login_type == "Member" else 0,
        1 if data.login_type == "New " else 0,
    ]
    X = align_features(features, churn_features)
    X_scaled = churn_scaler.transform(X)
    prediction = churn_model.predict(X_scaled)[0]
    return {"churn": int(prediction)}

@app.post("/predict/segment")
async def predict_segment(data: SegmentInput):
    features = [
        data.total_spent,
        data.recency,
        data.avg_profit,
        data.avg_discount,
        data.avg_quantity,
        data.order_count,
        1 if data.device_type == "Mobile" else 0,
        1 if data.device_type == "Web" else 0,
        1 if data.login_type == "First SignUp" else 0,
        1 if data.login_type == "Guest" else 0,
        1 if data.login_type == "Member" else 0,
        1 if data.login_type == "New " else 0,
    ]
    X = align_features(features, segment_features)
    X_scaled = segment_scaler.transform(X)
    prediction = segment_model.predict(X_scaled)[0]
    return {"segment_cluster": int(prediction)}

@app.post("/predict/fraud")
async def predict_fraud(data: FraudInput):
    features = [
        data.sales,
        data.quantity,
        data.discount,
        data.profit,
        data.shipping_cost,
        1 if data.device_type == "Mobile" else 0,
        1 if data.device_type == "Web" else 0,
        1 if data.login_type == "First SignUp" else 0,
        1 if data.login_type == "Guest" else 0,
        1 if data.login_type == "Member" else 0,
        1 if data.login_type == "New " else 0,
    ]
    X = align_features(features, fraud_features)
    X_scaled = fraud_scaler.transform(X)
    prediction = fraud_model.predict(X_scaled)[0]
    return {"fraud": int(prediction)}

@app.post("/recommend/products")
async def recommend_products(data: RecommendInput):
    customer_id = data.customer_id
    if customer_id not in user_product_matrix.index:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    distances, indices = similarity_knn.kneighbors(user_product_matrix.loc[[customer_id]], n_neighbors=6)
    similar_users = user_product_matrix.index[indices.flatten()[1:]]
    products = user_product_matrix.loc[similar_users].sum().sort_values(ascending=False)
    already_bought = user_product_matrix.loc[customer_id]
    recommendations = products[already_bought == 0].head(5).index.tolist()

    return {"recommended_products": recommendations}

@app.post("/predict/profit")
async def predict_profit(data: ProfitInput):
    features = [
        data.quantity,
        data.discount,
        data.sales,
        data.sales / data.quantity if data.quantity != 0 else 0,
        data.sales * (1 - data.discount)
    ]
    prediction = profit_model.predict([features])[0]
    return {"predicted_profit": float(prediction)}

@app.post("/predict/purchase_time")
async def predict_purchase_time(data: PurchaseTimeInput):
    features = [data.quantity, data.sales, data.sales]
    prediction = purchase_time_model.predict([features])[0]
    mapping = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}
    return {"predicted_purchase_time": mapping.get(int(prediction), "Unknown")}

@app.post("/predict/sales")
async def predict_sales(data: SalesInput):
    features = [
        data.aging,
        data.customer_id,
        data.product_category,
        data.product,
        data.quantity,
        data.discount,
        data.profit,
        data.shipping_cost,
        data.order_id,
        data.day,
        data.month,
        data.year
    ]
    X = align_features(features, sales_features)
    X[['Quantity', 'Discount', 'Shipping_Cost']] = sales_scaler.transform(X[['Quantity', 'Discount', 'Shipping_Cost']])
    prediction = sales_model.predict(X)[0]
    return {"predicted_sales": float(prediction)}

@app.get("/")
async def root():
    return {"message": "✨ E-commerce ML API Running Successfully!"}


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from utils import scale_input, recommend_products



app = FastAPI(title="E-Commerce Intelligence API")

# ========== Load Models ==========
churn_model = joblib.load("churn_rf_model.pkl")
churn_scaler = joblib.load("churn_scaler.pkl")

fraud_model = joblib.load("fraud_rf_model.pkl")
fraud_scaler = joblib.load("fraud_scaler.pkl")

profit_model = joblib.load("profit_prediction_model.pkl")

time_model = joblib.load("purchase_time_prediction_model.pkl")

product_matrix = joblib.load("user_product_matrix.pkl")
similarity_df = joblib.load("user_similarity.pkl")

# Load sales prediction model (use your best performing one)
sales_model = XGBRegressor()
sales_model.load_model("xgb_model.json")  # Or use joblib if saved like that

# If scaler is required (like for Quantity, Discount, Shipping_Cost), load it too:
sales_scaler = joblib.load("sales_scaler.pkl")  # optional


kmeans_model = joblib.load("customer_kmeans_model.pkl")
segmentation_scaler = joblib.load("customer_scaler.pkl")


# ========== Request Models ==========
class ChurnRequest(BaseModel):
    Total_Spent: float
    Recency: int
    Avg_Profit: float
    Avg_Discount: float
    Avg_Quantity: float
    Order_Count: int
    Device_Type_Desktop: int = 0
    Device_Type_Mobile: int = 0
    Customer_Login_type_Manual: int = 0
    Customer_Login_type_Social: int = 0

class FraudRequest(BaseModel):
    Sales: float
    Quantity: float
    Discount: float
    Profit: float
    Shipping_Cost: float
    Device_Type_Desktop: int = 0
    Device_Type_Mobile: int = 0
    Customer_Login_type_Manual: int = 0
    Customer_Login_type_Social: int = 0

class ProfitRequest(BaseModel):
    Quantity: float
    UnitPrice: float

class PurchaseTimeRequest(BaseModel):
    Quantity: float
    UnitPrice: float

class SalesRequest(BaseModel):
    Quantity: float
    Discount: float
    Shipping_Cost: float
    Product: int
    Product_Category: int
    Day: int
    Month: int
    Year: int
    
class CustomerSegmentRequest(BaseModel):
    Total_Spent: float
    Recency: int
    Avg_Profit: float
    Avg_Discount: float
    Avg_Quantity: float
    Order_Count: int
    Device_Type_Desktop: int = 0
    Device_Type_Mobile: int = 0
    Customer_Login_type_Manual: int = 0
    Customer_Login_type_Social: int = 0




# ========== Churn Endpoint ==========
@app.post("/predict/churn")
def predict_churn(data: ChurnRequest):
    try:
        df = pd.DataFrame([data.dict()])
        scaled = churn_scaler.transform(df)
        pred = churn_model.predict(scaled)[0]
        return {"churn_prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Fraud Detection ==========
@app.post("/predict/fraud")
def predict_fraud(data: FraudRequest):
    try:
        df = pd.DataFrame([data.dict()])
        scaled = fraud_scaler.transform(df)
        pred = fraud_model.predict(scaled)[0]
        return {"fraud_prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Profit Prediction ==========
@app.post("/predict/profit")
def predict_profit(data: ProfitRequest):
    try:
        df = pd.DataFrame([{
            "Quantity": data.Quantity,
            "UnitPrice": data.UnitPrice,
            "Total_Sale": data.Quantity * data.UnitPrice
        }])
        pred = profit_model.predict(df)[0]
        return {"predicted_profit": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Purchase Time Prediction ==========
@app.post("/predict/purchase-time")
def predict_time(data: PurchaseTimeRequest):
    try:
        total_amount = data.Quantity * data.UnitPrice
        df = pd.DataFrame([{
            "Quantity": data.Quantity,
            "UnitPrice": data.UnitPrice,
            "TotalAmount": total_amount
        }])
        pred = time_model.predict(df)[0]
        time_bins = ['Morning', 'Afternoon', 'Evening', 'Night']
        return {"predicted_time_bin": time_bins[pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Product Recommendations ==========
@app.get("/recommend/products")
def recommend_products(customer_id: str):
    try:
        if customer_id not in product_matrix.index:
            return {"error": f"Customer ID {customer_id} not found."}
        similar_users = similarity_df[customer_id].sort_values(ascending=False)[1:6].index
        product_scores = product_matrix.loc[similar_users].sum().sort_values(ascending=False)
        already_bought = product_matrix.loc[customer_id]
        recommendations = product_scores[already_bought == 0].head(5).index.tolist()
        return {"recommended_products": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict/sales")
def predict_sales(data: SalesRequest):
    try:
        df = pd.DataFrame([data.dict()])
        # Optionally scale some features
        df[['Quantity', 'Discount', 'Shipping_Cost']] = sales_scaler.transform(df[['Quantity', 'Discount', 'Shipping_Cost']])
        prediction = sales_model.predict(df)[0]
        return {"predicted_sales": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/customers")
def segment_customer(data: CustomerSegmentRequest):
    try:
        df = pd.DataFrame([data.dict()])
        scaled = segmentation_scaler.transform(df)
        cluster = kmeans_model.predict(scaled)[0]
        return {"customer_segment": int(cluster)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


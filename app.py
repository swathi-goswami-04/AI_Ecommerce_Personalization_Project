import streamlit as st
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# --- Page setup ---
st.set_page_config(page_title="🚀 E-commerce AI Platform", layout="wide")

# --- Load Models ---
@st.cache_resource
def load_models():
    models = {}
    models['churn_model'] = joblib.load("churn_rf_model.pkl")
    models['churn_scaler'] = joblib.load("churn_scaler.pkl")
    models['churn_features'] = joblib.load("churn_features.pkl")

    models['segment_model'] = joblib.load("customer_kmeans_model.pkl")
    models['segment_scaler'] = joblib.load("customer_scaler.pkl")
    models['segment_features'] = joblib.load("segment_features.pkl")

    models['fraud_model'] = joblib.load("fraud_rf_model.pkl")
    models['fraud_scaler'] = joblib.load("fraud_scaler.pkl")
    models['fraud_features'] = joblib.load("fraud_features.pkl")

    models['user_product_matrix'] = joblib.load("user_product_matrix.pkl")
    models['user_knn_model'] = joblib.load("user_knn_model.pkl")

    models['profit_model'] = joblib.load("profit_prediction_model.pkl")
    models['purchase_time_model'] = joblib.load("purchase_time_prediction_model.pkl")

    sales_model = XGBRegressor()
    sales_model.load_model("xgb_model.json")
    models['sales_model'] = sales_model
    models['sales_scaler'] = joblib.load("sales_scaler.pkl")
    models['sales_features'] = joblib.load("sales_features.pkl")

    return models

models = load_models()

# --- Helper ---
def align_features(input_data, feature_list):
    df = pd.DataFrame([input_data], columns=feature_list)
    for feature in feature_list:
        if feature not in df.columns:
            df[feature] = 0
    df = df[feature_list]
    return df

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a Module", [
    "Churn Prediction",
    "Customer Segmentation",
    "Fraud Detection",
    "Product Recommendation",
    "Profit Prediction",
    "Purchase Time Prediction",
    "Sales Prediction"
])

st.title("🚀 E-commerce AI Personalization Suite")

# --- Pages ---
if page == "Churn Prediction":
    st.header("🔮 Customer Churn Prediction")
    total_spent = st.number_input("Total Spent", value=0.0)
    recency = st.number_input("Recency (Days)", value=0)
    avg_profit = st.number_input("Average Profit", value=0.0)
    avg_discount = st.number_input("Average Discount", value=0.0)
    avg_quantity = st.number_input("Average Quantity", value=0.0)
    order_count = st.number_input("Order Count", value=0)
    device_type = st.selectbox("Device Type", ["Mobile", "Web"])
    login_type = st.selectbox("Login Type", ["First SignUp", "Guest", "Member", "New "])

    if st.button("Predict Churn"):
        features = [
            total_spent, recency, avg_profit, avg_discount, avg_quantity, order_count,
            1 if device_type == "Mobile" else 0,
            1 if device_type == "Web" else 0,
            1 if login_type == "First SignUp" else 0,
            1 if login_type == "Guest" else 0,
            1 if login_type == "Member" else 0,
            1 if login_type == "New " else 0,
        ]
        X = align_features(features, models['churn_features'])
        X_scaled = models['churn_scaler'].transform(X)
        pred = models['churn_model'].predict(X_scaled)[0]
        st.success(f"🎯 Customer Status: {'Churned' if pred == 1 else 'Active'}")

elif page == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    total_spent = st.number_input("Total Spent", value=0.0)
    recency = st.number_input("Recency (Days)", value=0)
    avg_profit = st.number_input("Average Profit", value=0.0)
    avg_discount = st.number_input("Average Discount", value=0.0)
    avg_quantity = st.number_input("Average Quantity", value=0.0)
    order_count = st.number_input("Order Count", value=0)
    device_type = st.selectbox("Device Type", ["Mobile", "Web"])
    login_type = st.selectbox("Login Type", ["First SignUp", "Guest", "Member", "New "])

    if st.button("Segment Customer"):
        features = [
            total_spent, recency, avg_profit, avg_discount, avg_quantity, order_count,
            1 if device_type == "Mobile" else 0,
            1 if device_type == "Web" else 0,
            1 if login_type == "First SignUp" else 0,
            1 if login_type == "Guest" else 0,
            1 if login_type == "Member" else 0,
            1 if login_type == "New " else 0,
        ]
        X = align_features(features, models['segment_features'])
        X_scaled = models['segment_scaler'].transform(X)
        pred = models['segment_model'].predict(X_scaled)[0]
        st.success(f"🎯 Customer Segment: Cluster {pred}")

elif page == "Fraud Detection":
    st.header("🚨 Fraud Detection")
    sales = st.number_input("Sales", value=0.0)
    quantity = st.number_input("Quantity", value=0)
    discount = st.number_input("Discount", value=0.0)
    profit = st.number_input("Profit", value=0.0)
    shipping_cost = st.number_input("Shipping Cost", value=0.0)
    device_type = st.selectbox("Device Type", ["Mobile", "Web"])
    login_type = st.selectbox("Login Type", ["First SignUp", "Guest", "Member", "New "])

    if st.button("Detect Fraud"):
        features = [
            sales, quantity, discount, profit, shipping_cost,
            1 if device_type == "Mobile" else 0,
            1 if device_type == "Web" else 0,
            1 if login_type == "First SignUp" else 0,
            1 if login_type == "Guest" else 0,
            1 if login_type == "Member" else 0,
            1 if login_type == "New " else 0,
        ]
        X = align_features(features, models['fraud_features'])
        X_scaled = models['fraud_scaler'].transform(X)
        pred = models['fraud_model'].predict(X_scaled)[0]
        st.success(f"⚡ Fraud Prediction: {'Fraudulent' if pred == 1 else 'Legit'}")

elif page == "Product Recommendation":
    st.header("🎁 Product Recommendation")
    customer_id = st.number_input("Customer ID", min_value=1, value=1)

    if st.button("Recommend Products"):
        if customer_id not in models['user_product_matrix'].index:
            st.error("❌ Customer not found!")
        else:
            distances, indices = models['user_knn_model'].kneighbors(models['user_product_matrix'].loc[[customer_id]], n_neighbors=6)
            similar_users = models['user_product_matrix'].index[indices.flatten()[1:]]
            products = models['user_product_matrix'].loc[similar_users].sum().sort_values(ascending=False)
            already_bought = models['user_product_matrix'].loc[customer_id]
            recs = products[already_bought == 0].head(5).index.tolist()
            st.success(f"🎯 Recommended Products: {recs}")

elif page == "Profit Prediction":
    st.header("💸 Profit Prediction")
    quantity = st.number_input("Quantity", value=1.0)
    discount = st.number_input("Discount", value=0.0)
    sales = st.number_input("Sales", value=1.0)

    if st.button("Predict Profit"):
        features = [
            quantity,
            discount,
            sales,
            sales / quantity if quantity != 0 else 0,
            sales * (1 - discount)
        ]
        pred = models['profit_model'].predict([features])[0]
        st.success(f"💵 Predicted Profit: {pred:.2f}")

elif page == "Purchase Time Prediction":
    st.header("🕒 Purchase Time Prediction")
    quantity = st.number_input("Quantity", value=1.0)
    sales = st.number_input("Sales", value=1.0)

    if st.button("Predict Purchase Time"):
        features = [quantity, sales, sales]
        pred = models['purchase_time_model'].predict([features])[0]
        mapping = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}
        st.success(f"🕰️ Predicted Purchase Time: {mapping.get(int(pred), 'Unknown')}")

elif page == "Sales Prediction":
    st.header("📈 Sales Prediction")
    aging = st.number_input("Aging", value=1.0)
    customer_id = st.number_input("Customer ID", value=0)
    product_category = st.number_input("Product Category", value=0)
    product = st.number_input("Product", value=0)
    quantity = st.number_input("Quantity", value=0)
    discount = st.number_input("Discount", value=0.0)
    profit = st.number_input("Profit", value=0.0)
    shipping_cost = st.number_input("Shipping Cost", value=0.0)
    order_id = st.number_input("Order ID", value=0)
    day = st.number_input("Day", value=1)
    month = st.number_input("Month", value=1)
    year = st.number_input("Year", value=2024)

    if st.button("Predict Sales"):
        features = [
            aging, customer_id, product_category, product,
            quantity, discount, profit, shipping_cost,
            order_id, day, month, year
        ]
        X = align_features(features, models['sales_features'])
        X[['Quantity', 'Discount', 'Shipping_Cost']] = models['sales_scaler'].transform(X[['Quantity', 'Discount', 'Shipping_Cost']])
        pred = models['sales_model'].predict(X)[0]
        st.success(f"📈 Predicted Sales: {pred:.2f}")

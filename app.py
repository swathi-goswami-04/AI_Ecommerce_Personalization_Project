import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="E-Commerce AI Suite", layout="wide")
st.title("🛍️ E-Commerce Intelligence Dashboard")

api_url = "http://localhost:8000"

option = st.sidebar.selectbox("Choose a Prediction Module or Chart", (
    "Churn Prediction",
    "Fraud Detection",
    "Profit Prediction",
    "Purchase Time Prediction",
    "Product Recommendation",
    "Sales Prediction",
    "Customer Segmentation",
    "📊 Visualize Insights"
))

if option == "Churn Prediction":
    st.subheader("📉 Predict Customer Churn")
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_spent = st.number_input("Total Spent", min_value=0.0)
            recency = st.number_input("Recency (days)", min_value=0)
            avg_profit = st.number_input("Average Profit")
            avg_discount = st.number_input("Average Discount")
            avg_quantity = st.number_input("Average Quantity")
            order_count = st.number_input("Order Count", min_value=0)
        with col2:
            device = st.selectbox("Device Type", ["Desktop", "Mobile"])
            login = st.selectbox("Login Type", ["Manual", "Social"])
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        data = {
            "Total_Spent": total_spent,
            "Recency": recency,
            "Avg_Profit": avg_profit,
            "Avg_Discount": avg_discount,
            "Avg_Quantity": avg_quantity,
            "Order_Count": order_count,
            f"Device_Type_{device}": 1,
            f"Customer_Login_type_{login}": 1
        }
        response = requests.post(f"{api_url}/predict/churn", json=data)
        st.success(f"🔮 Churn Prediction: {'Yes' if response.json()['churn_prediction'] else 'No'}")

elif option == "Fraud Detection":
    st.subheader("🚨 Predict Fraudulent Transaction")
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            sales = st.number_input("Sales")
            quantity = st.number_input("Quantity")
            discount = st.number_input("Discount")
            profit = st.number_input("Profit")
            shipping = st.number_input("Shipping Cost")
        with col2:
            device = st.selectbox("Device Type", ["Desktop", "Mobile"])
            login = st.selectbox("Login Type", ["Manual", "Social"])
        submitted = st.form_submit_button("Detect Fraud")

    if submitted:
        data = {
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "Shipping_Cost": shipping,
            f"Device_Type_{device}": 1,
            f"Customer_Login_type_{login}": 1
        }
        response = requests.post(f"{api_url}/predict/fraud", json=data)
        result = response.json()["fraud_prediction"]
        st.warning("⚠️ Fraud Detected!" if result else "✅ Transaction is Safe")

elif option == "Profit Prediction":
    st.subheader("💸 Predict Profit")
    quantity = st.number_input("Quantity")
    unit_price = st.number_input("Unit Price")
    if st.button("Predict Profit"):
        response = requests.post(f"{api_url}/predict/profit", json={
            "Quantity": quantity,
            "UnitPrice": unit_price
        })
        st.success(f"💰 Predicted Profit: {response.json()['predicted_profit']}")

elif option == "Purchase Time Prediction":
    st.subheader("🕓 Predict Purchase Time")
    quantity = st.number_input("Quantity")
    unit_price = st.number_input("Unit Price")
    if st.button("Predict Time"):
        response = requests.post(f"{api_url}/predict/purchase-time", json={
            "Quantity": quantity,
            "UnitPrice": unit_price
        })
        st.success(f"🕰️ Most Likely Time: {response.json()['predicted_time_bin']}")

elif option == "Product Recommendation":
    st.subheader("🎁 Recommend Products for Customer")
    customer_id = st.text_input("Customer ID")
    if st.button("Get Recommendations"):
        response = requests.get(f"{api_url}/recommend/products", params={"customer_id": customer_id})
        if "error" in response.json():
            st.error(response.json()["error"])
        else:
            st.success("🎯 Top Recommendations:")
            st.write(response.json()["recommended_products"])

elif option == "Sales Prediction":
    st.subheader("📈 Predict Sales")
    with st.form("sales_form"):
        quantity = st.number_input("Quantity")
        discount = st.number_input("Discount")
        shipping = st.number_input("Shipping Cost")
        product = st.number_input("Product Code")
        category = st.number_input("Category Code")
        day = st.number_input("Day")
        month = st.number_input("Month")
        year = st.number_input("Year")
        submit = st.form_submit_button("Predict Sales")

    if submit:
        response = requests.post(f"{api_url}/predict/sales", json={
            "Quantity": quantity,
            "Discount": discount,
            "Shipping_Cost": shipping,
            "Product": product,
            "Product_Category": category,
            "Day": day,
            "Month": month,
            "Year": year
        })
        st.success(f"📊 Predicted Sales: {response.json()['predicted_sales']}")

elif option == "Customer Segmentation":
    st.subheader("🔍 Segment a Customer")
    with st.form("segment_form"):
        total_spent = st.number_input("Total Spent")
        recency = st.number_input("Recency (days)")
        avg_profit = st.number_input("Average Profit")
        avg_discount = st.number_input("Average Discount")
        avg_quantity = st.number_input("Average Quantity")
        order_count = st.number_input("Order Count")
        device = st.selectbox("Device Type", ["Desktop", "Mobile"])
        login = st.selectbox("Login Type", ["Manual", "Social"])
        submitted = st.form_submit_button("Segment Customer")

    if submitted:
        data = {
            "Total_Spent": total_spent,
            "Recency": recency,
            "Avg_Profit": avg_profit,
            "Avg_Discount": avg_discount,
            "Avg_Quantity": avg_quantity,
            "Order_Count": order_count,
            f"Device_Type_{device}": 1,
            f"Customer_Login_type_{login}": 1
        }
        response = requests.post(f"{api_url}/segment/customers", json=data)
        st.success(f"📌 Assigned Segment: {response.json()['customer_segment']}")

elif option == "📊 Visualize Insights":
    st.subheader("📊 Business Insights Dashboard")

    df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    chart = st.selectbox("Choose a Chart", (
        "Sales Over Time",
        "Top Product Categories",
        "Churn Distribution (Recency > 90 days)",
        "Fraud vs Safe Transactions"
    ))

    if chart == "Sales Over Time":
        st.write("### 💵 Total Sales Over Time")
        df_daily = df.groupby(df['Order_Date'].dt.date)['Sales'].sum().reset_index()
        st.line_chart(df_daily.set_index('Order_Date'))

    elif chart == "Top Product Categories":
        st.write("### 📦 Top-Selling Product Categories")
        cat_sales = df.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(cat_sales)

    elif chart == "Churn Distribution (Recency > 90 days)":
        st.write("### 🧍‍♀️ Churned vs Active Customers")
        today = pd.Timestamp.today()
        churn_df = df.groupby('Customer_Id')['Order_Date'].max().reset_index()
        churn_df['Recency'] = (today - churn_df['Order_Date']).dt.days
        churn_df['Churn'] = churn_df['Recency'].apply(lambda x: 1 if x > 90 else 0)
        plt.figure(figsize=(4,4))
        plt.pie(churn_df['Churn'].value_counts(), labels=['Active', 'Churned'],
                autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.axis('equal')
        st.pyplot(plt)

    elif chart == "Fraud vs Safe Transactions":
        st.write("### 🔍 Fraud Detection (Rule-Based)")
        df['Is_Fraud'] = df.apply(lambda row: 1 if (row['Discount'] > 0.5 and row['Quantity'] > 5 and row['Profit'] < 0) else 0, axis=1)
        fraud_counts = df['Is_Fraud'].value_counts()
        labels = ['Safe', 'Fraud']
        fig, ax = plt.subplots()
        ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=["#8BC34A", "#E53935"])
        ax.axis('equal')
        st.pyplot(fig)

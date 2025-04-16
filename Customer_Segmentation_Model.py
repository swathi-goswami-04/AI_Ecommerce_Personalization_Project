import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')

# Convert date
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

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

# Add most common device & login type per customer
extra_features = df.groupby('Customer_Id').agg({
    'Device_Type': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'Customer_Login_type': lambda x: x.mode()[0] if not x.mode().empty else np.nan
}).reset_index()

customer_df = customer_df.merge(extra_features, on='Customer_Id', how='left')

# One-hot encode categorical vars
customer_df = pd.get_dummies(customer_df, columns=['Device_Type', 'Customer_Login_type'])

# Keep ID separate
customer_ids = customer_df['Customer_Id']
X = customer_df.drop(['Customer_Id'], axis=1)

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow Method to determine K ---
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.savefig("elbow_plot.png")
plt.show()

# Choose K (manually or based on elbow)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# --- Add Cluster Info ---
customer_df['Cluster'] = clusters

# --- PCA for Visualization ---
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis', s=50)
plt.title('Customer Segments (PCA Visual)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title="Cluster")
plt.grid()
plt.savefig("customer_segments_pca.png")
plt.show()

# --- Save Models ---
joblib.dump(kmeans, "customer_kmeans_model.pkl")
joblib.dump(scaler, "customer_scaler.pkl")
joblib.dump(pca, "customer_pca.pkl")

# --- Save Segmented Customers ---
customer_df.to_csv("segmented_customers.csv", index=False)

print("🎯 Customer segmentation complete. Models and outputs saved!")

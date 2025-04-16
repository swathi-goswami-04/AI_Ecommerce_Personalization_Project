import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load dataset
df = pd.read_csv("E_commerce_Dataset.csv", encoding='latin1')

# --- Preprocessing ---
# Drop duplicates and NaNs just in case
df = df.dropna(subset=['Customer_ID', 'Product_Name'])

# Create a user-product matrix
user_product_matrix = df.pivot_table(
    index='Customer_ID',
    columns='Product_Name',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# --- Cosine Similarity ---
cosine_sim = cosine_similarity(user_product_matrix)
similarity_df = pd.DataFrame(cosine_sim, index=user_product_matrix.index, columns=user_product_matrix.index)

# --- Recommendation Function ---
def recommend_products(customer_id, top_n=5):
    if customer_id not in user_product_matrix.index:
        return f"🛑 Customer ID {customer_id} not found!"
    
    similar_users = similarity_df[customer_id].sort_values(ascending=False)[1:]  # Skip self
    top_similar_users = similar_users.head(5).index

    # Get products bought by similar users
    product_scores = user_product_matrix.loc[top_similar_users].sum().sort_values(ascending=False)

    # Remove already purchased items
    user_purchased = user_product_matrix.loc[customer_id]
    product_scores = product_scores[user_purchased == 0]

    # Top N recommendations
    recommendations = product_scores.head(top_n).index.tolist()
    return recommendations

# --- Test the function ---
sample_customer = df['Customer_ID'].iloc[0]
recs = recommend_products(sample_customer)
print(f"🎁 Recommended products for Customer {sample_customer}:\n{recs}")

# Save recommendation artifacts
joblib.dump(user_product_matrix, "user_product_matrix.pkl")
joblib.dump(similarity_df, "user_similarity.pkl")

print("✅ Product recommendation model created and saved.")

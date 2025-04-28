import pandas as pd
import numpy as np
import joblib
from typing import Dict, List



def scale_input(data_dict, scaler):
    df = pd.DataFrame([data_dict])
    return scaler.transform(df)

def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def recommend_products(customer_id, product_matrix, similarity_df, top_n=5):
    if customer_id not in product_matrix.index:
        return []
    similar_users = similarity_df[customer_id].sort_values(ascending=False)[1:6].index
    product_scores = product_matrix.loc[similar_users].sum().sort_values(ascending=False)
    already_bought = product_matrix.loc[customer_id]
    recommendations = product_scores[already_bought == 0].head(top_n).index.tolist()
    return recommendations

# Data loading functions

import pandas as pd
import os

RAW_DATA_PATH = "data/raw/"

def load_olist_data():
    """
    Loads all Olist CSV files from data/raw/ and returns them as a dictionary.
    """
    files = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "category_translation": "product_category_name_translation.csv"
    }

    data = {}
    for key, filename in files.items():
        path = os.path.join(RAW_DATA_PATH, filename)
        data[key] = pd.read_csv(path)

    print("All raw Olist datasets loaded successfully.")
    return data


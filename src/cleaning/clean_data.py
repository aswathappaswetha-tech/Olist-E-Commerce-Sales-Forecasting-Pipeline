
import pandas as pd

def clean_olist_data(data_dict):
    """
    Cleans and merges Olist datasets into a single dataframe.
    Expects a dictionary of raw dataframes loaded by ingestion.
    """

    # Extract tables
    customers = data_dict["customers"]
    orders = data_dict["orders"]
    order_items = data_dict["order_items"]
    products = data_dict["products"]
    sellers = data_dict["sellers"]
    payments = data_dict["payments"]
    reviews = data_dict["reviews"]

    # Convert timestamps to datetime
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for col in timestamp_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Basic cleaning
    customers = customers.drop_duplicates()
    orders = orders.drop_duplicates()
    order_items = order_items.drop_duplicates()
    products = products.drop_duplicates()
    sellers = sellers.drop_duplicates()
    payments = payments.drop_duplicates()
    reviews = reviews.drop_duplicates()

    # Merge tables
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(order_items, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(payments, on="order_id", how="left")
    df = df.merge(reviews, on="order_id", how="left")

    print("Cleaning and merging completed.")
    return df

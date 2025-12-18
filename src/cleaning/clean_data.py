# Data cleaning functions
"""
Data Ingestion Module
---------------------
This module loads raw CSV files from the Olist dataset and returns
clean Pandas DataFrames for further processing.
"""

import os
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded file: {file_path} | Shape: {df.shape}")
    return df


def load_olist_dataset(raw_data_dir: str) -> dict:
    """
    Loads all Olist dataset CSVs into a dictionary of DataFrames.

    Parameters
    ----------
    raw_data_dir : str
        Directory containing raw Olist CSV files.

    Returns
    -------
    dict
        Dictionary of DataFrames keyed by table name.
    """

    files = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv"
    }

    dataset = {}

    for key, filename in files.items():
        path = os.path.join(raw_data_dir, filename)
        dataset[key] = load_csv(path)

    print("\nAll Olist tables loaded successfully.")
    return dataset
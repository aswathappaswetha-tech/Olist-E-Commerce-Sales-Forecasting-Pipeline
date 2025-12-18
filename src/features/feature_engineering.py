# Feature engineering functions
"""
Feature Engineering Module
--------------------------
This module creates time-based, lag-based, rolling, customer-level,
and product-level features for the Olist dataset.

The output is a modeling-ready dataset aggregated at daily level.
"""

import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "order_purchase_timestamp") -> pd.DataFrame:
    """
    Adds calendar-based features from the purchase timestamp.
    """
    df["order_date"] = df[date_col].dt.date
    df["order_month"] = df[date_col].dt.month
    df["order_week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["order_dayofweek"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    return df


def aggregate_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates total daily revenue from order items.
    """

    df = df.copy()

    daily = (
        df.groupby("order_date")["price"]
        .sum()
        .reset_index()
    )

    # Rename to Prophet format
    daily.rename(columns={"order_date": "ds", "price": "y"}, inplace=True)

    # Ensure correct types
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily["y"] = pd.to_numeric(daily["y"])

    return daily


def add_lag_features(df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
    """
    Adds lag features for time series modeling.
    Expects the target column to be 'y'.
    """

    df = df.copy()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")

    df["lag_1"] = df[target_col].shift(1)
    df["lag_7"] = df[target_col].shift(7)
    df["lag_30"] = df[target_col].shift(30)

    return df

def add_rolling_features(df: pd.DataFrame, target_col: str = "daily_sales") -> pd.DataFrame:
    """
    Adds rolling window statistics.
    """
    df["rolling_7_mean"] = df[target_col].rolling(7).mean()
    df["rolling_30_mean"] = df[target_col].rolling(30).mean()
    df["rolling_90_mean"] = df[target_col].rolling(90).mean()
    return df


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds customer-level metrics.
    """
    customer_stats = (
        df.groupby("customer_id")
        .agg(
            customer_total_orders=("order_id", "nunique"),
            customer_avg_price=("price", "mean"),
        )
        .reset_index()
    )
    df = df.merge(customer_stats, on="customer_id", how="left")
    return df


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds product-level metrics.
    """
    product_stats = (
        df.groupby("product_id")
        .agg(
            product_total_sales=("price", "sum"),
            product_avg_price=("price", "mean"),
            product_order_count=("order_id", "nunique"),
        )
        .reset_index()
    )
    df = df.merge(product_stats, on="product_id", how="left")
    return df

def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df_clean)
    daily = aggregate_daily_revenue(df)
    daily = add_lag_features(daily, target_col="y")
    print(f"Feature engineering complete. Final shape: {daily.shape}")
    return daily

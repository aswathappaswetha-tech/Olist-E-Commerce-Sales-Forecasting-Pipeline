# Modeling functions
"""
Prophet Modeling Module
-----------------------
This module trains a Prophet forecasting model on daily sales data,
generates forecasts, and evaluates model performance.
"""

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet requires columns: ds (date) and y (target).
    """
    prophet_df = df.rename(columns={"order_date": "ds", "daily_sales": "y"})
    prophet_df = prophet_df[["ds", "y"]].dropna()
    return prophet_df


def train_test_split(df: pd.DataFrame, test_days: int = 60):
    """
    Splits the dataset into train and test sets.
    """
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test


def train_prophet_model(train_df: pd.DataFrame) -> Prophet:
    """
    Trains a Prophet model.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive"
    )
    model.fit(train_df)
    return model


def forecast_sales(model: Prophet, periods: int = 60) -> pd.DataFrame:
    """
    Generates future predictions.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    """
    Computes MAE, RMSE, and MAPE.
    """
    merged = test_df.merge(forecast_df[["ds", "yhat"]], on="ds", how="left")

    mae = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
    mape = np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"])) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_prophet_pipeline(df: pd.DataFrame, test_days: int = 60):
    """
    Full modeling pipeline.
    """
    prophet_df = prepare_prophet_data(df)
    train_df, test_df = train_test_split(prophet_df, test_days)

    model = train_prophet_model(train_df)
    forecast_df = forecast_sales(model, periods=test_days)

    metrics = evaluate_forecast(test_df, forecast_df)

    print("Forecasting complete.")
    print("Evaluation metrics:", metrics)

    return forecast_df, metrics
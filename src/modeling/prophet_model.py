"""
Prophet Modeling Module
-----------------------
This module prepares data for Prophet, trains the model,
generates forecasts, and evaluates performance.
"""

import pandas as pd
from prophet import Prophet


# ---------------------------------------------------------
# 1. Prepare data for Prophet
# ---------------------------------------------------------

def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for Prophet using the feature-engineered dataframe.
    Expects columns: ds (datetime), y (target).
    """

    df = df.copy()

    if "ds" not in df.columns:
        raise KeyError("Feature engineering did not produce 'ds'. Check feature_engineering.py.")

    if "y" not in df.columns:
        raise KeyError("Feature engineering did not produce 'y'. Check feature_engineering.py.")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    return df[["ds", "y"]]

# ---------------------------------------------------------
# 2. Train/Test Split
# ---------------------------------------------------------
def train_test_split_prophet(df: pd.DataFrame, test_days: int = 30):
    """
    Splits Prophet dataframe into train and test sets.
    """
    df = df.sort_values("ds")

    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]

    return train, test


# ---------------------------------------------------------
# 3. Train Prophet Model
# ---------------------------------------------------------
def train_prophet_model(train_df: pd.DataFrame) -> Prophet:
    """
    Trains a Prophet model on the training dataframe.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_df)
    return model


# ---------------------------------------------------------
# 4. Forecast Future Values
# ---------------------------------------------------------
def generate_forecast(model: Prophet, periods: int = 30) -> pd.DataFrame:
    """
    Generates a forecast for the next `periods` days.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


# ---------------------------------------------------------
# 5. Evaluate Forecast
# ---------------------------------------------------------
def evaluate_forecast(test_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """
    Merges test data with forecast and computes MAE and RMSE.
    """

    # Ensure datetime alignment
    test_df["ds"] = pd.to_datetime(test_df["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # Merge on ds
    merged = test_df.merge(
        forecast_df[["ds", "yhat"]],
        on="ds",
        how="left"
    )

    merged["error"] = merged["y"] - merged["yhat"]

    mae = merged["error"].abs().mean()
    rmse = (merged["error"] ** 2).mean() ** 0.5

    return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------
# 6. Full Pipeline Wrapper
# ---------------------------------------------------------
def run_prophet_pipeline(df_features: pd.DataFrame):
    """
    Full end‑to‑end Prophet pipeline:
    - Prepare data
    - Train/test split
    - Train model
    - Forecast
    - Evaluate
    """

    print("Preparing data for Prophet...")
    prophet_df = prepare_prophet_data(df_features)

    print("Splitting train/test...")
    train_df, test_df = train_test_split_prophet(prophet_df)

    print("Training Prophet model...")
    model = train_prophet_model(train_df)

    print("Generating forecast...")
    forecast_df = generate_forecast(model)

    print("Evaluating forecast...")
    metrics = evaluate_forecast(test_df, forecast_df)

    print("Forecasting complete.")
    print(f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

    return forecast_df, metrics



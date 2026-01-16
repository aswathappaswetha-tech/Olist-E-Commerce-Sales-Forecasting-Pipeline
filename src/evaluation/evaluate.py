import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    """
    Computes MAE and RMSE for forecast evaluation.
    """

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    mae = (y_true - y_pred).abs().mean()
    rmse = ((y_true - y_pred) ** 2).mean() ** 0.5

    return {"mae": mae, "rmse": rmse}


def merge_forecast(test_df, forecast_df):
    """
    Aligns test and forecast data on 'ds' and returns merged dataframe.
    Expects:
    - test_df: columns ['ds', 'y']
    - forecast_df: Prophet output with ['ds', 'yhat']
    """

    test_df = test_df.copy()
    forecast_df = forecast_df.copy()

    test_df["ds"] = pd.to_datetime(test_df["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    merged = test_df.merge(
        forecast_df[["ds", "yhat"]],
        on="ds",
        how="left"
    )

    return merged


def run_evaluation_plots(test_df, forecast_df):
    """
    Plots actual vs forecasted values using Prophet output.
    Expects:
    - test_df: columns ['ds', 'y']
    - forecast_df: Prophet forecast with ['ds', 'yhat']
    """

    merged = merge_forecast(test_df, forecast_df)

    plt.figure(figsize=(12, 6))
    plt.plot(merged["ds"], merged["y"], label="Actual", linewidth=2)
    plt.plot(merged["ds"], merged["yhat"], label="Forecast", linewidth=2)

    plt.title("Actual vs Forecasted Daily Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import pandas as pd

def baseline_forecast(df, test_days=30):
    df = df.copy()
    df["y_pred"] = df["y"].shift(1)
    test = df.tail(test_days)
    return test["y"], test["y_pred"]

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(df, test_days=30, order=(1,1,1), seasonal=(1,1,1,7)):
    train = df[:-test_days]
    test = df[-test_days:]

    model = SARIMAX(train["y"], order=order, seasonal_order=seasonal)
    results = model.fit(disp=False)

    forecast = results.predict(start=test.index[0], end=test.index[-1])
    return test["y"], forecast

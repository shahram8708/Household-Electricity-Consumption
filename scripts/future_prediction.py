
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def future_forecasting(data, steps=30):
    sarima_model = SARIMAX(data['Global_active_power'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit()

    future_forecast = sarima_result.forecast(steps=steps)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Global_active_power'], label='Historical Data')
    plt.plot(pd.date_range(data.index[-1], periods=steps, freq='D'), future_forecast, label='Future Forecast')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title('Future Electricity Consumption Forecast')
    plt.legend()
    plt.savefig('../results/plots/future_forecast.png')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('../data/feature_engineered_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    future_forecasting(data)


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def sarima_forecasting(data):
    train_data, test_data = train_test_split(data['Global_active_power'], test_size=0.2, shuffle=False)

    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit()

    sarima_forecast = sarima_result.forecast(steps=len(test_data))

    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data, label='Train')
    plt.plot(test_data.index, test_data, label='Test')
    plt.plot(test_data.index, sarima_forecast, label='SARIMA Forecast')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title('SARIMA Model Forecast')
    plt.legend()
    plt.savefig('../results/plots/sarima_forecast.png')
    plt.show()

    sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
    print(f'SARIMA RMSE: {sarima_rmse}')

if __name__ == "__main__":
    data = pd.read_csv('../data/preprocessed_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    sarima_forecasting(data)

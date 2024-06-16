
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def arima_forecasting(data):
    train_data, test_data = train_test_split(data['Global_active_power'], test_size=0.2, shuffle=False)

    arima_model = ARIMA(train_data, order=(5, 1, 0))
    arima_result = arima_model.fit()

    arima_forecast = arima_result.forecast(steps=len(test_data))

    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data, label='Train')
    plt.plot(test_data.index, test_data, label='Test')
    plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title('ARIMA Model Forecast')
    plt.legend()
    plt.savefig('../results/plots/arima_forecast.png')
    plt.show()

    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
    print(f'ARIMA RMSE: {arima_rmse}')

if __name__ == "__main__":
    data = pd.read_csv('../data/preprocessed_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    arima_forecasting(data)

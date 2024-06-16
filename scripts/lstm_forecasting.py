
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def create_lstm_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def lstm_forecasting(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Global_active_power'].values.reshape(-1,1))

    time_step = 10
    X, Y = create_lstm_dataset(scaled_data, time_step)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    lstm_model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=2)

    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(Y_test):], scaler.inverse_transform(Y_test.reshape(-1, 1)), label='Test')
    plt.plot(data.index[-len(lstm_predictions):], lstm_predictions, label='LSTM Forecast')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title('LSTM Model Forecast')
    plt.legend()
    plt.savefig('../results/plots/lstm_forecast.png')
    plt.show()

    lstm_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(Y_test.reshape(-1, 1)), lstm_predictions))
    print(f'LSTM RMSE: {lstm_rmse}')

if __name__ == "__main__":
    data = pd.read_csv('../data/preprocessed_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    lstm_forecasting(data)

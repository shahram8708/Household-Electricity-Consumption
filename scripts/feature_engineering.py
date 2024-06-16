
import pandas as pd

def add_features(data):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter

    return data

if __name__ == "__main__":
    data = pd.read_csv('../data/preprocessed_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    data = add_features(data)
    data.to_csv('../data/feature_engineered_household_power_consumption.csv')

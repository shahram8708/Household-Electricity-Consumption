
import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=';', 
                       parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True,
                       low_memory=False, na_values=['nan','?'])

    data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')
    data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'], errors='coerce')
    data['Voltage'] = pd.to_numeric(data['Voltage'], errors='coerce')
    data['Global_intensity'] = pd.to_numeric(data['Global_intensity'], errors='coerce')
    data['Sub_metering_1'] = pd.to_numeric(data['Sub_metering_1'], errors='coerce')
    data['Sub_metering_2'] = pd.to_numeric(data['Sub_metering_2'], errors='coerce')
    data['Sub_metering_3'] = pd.to_numeric(data['Sub_metering_3'], errors='coerce')

    data.dropna(inplace=True)

    data.set_index('datetime', inplace=True)

    return data

if __name__ == "__main__":
    data = preprocess_data('../data/household_power_consumption.txt')
    data.to_csv('../data/preprocessed_household_power_consumption.csv')

# Household Electricity Consumption Forecasting

This project focuses on forecasting household electricity consumption using time series analysis techniques in Python. The objective is to build and compare various models to accurately predict future electricity consumption trends based on historical data. The insights derived from this analysis aim to empower households to optimize energy usage, plan efficiently, and contribute to sustainable energy practices.

## Dataset Description

The dataset used includes the following features:

- **Datetime**: Date and time of the electricity consumption recording.
- **Global_active_power**: Total active power consumed by the household.
- **Global_reactive_power**: Total reactive power consumed by the household.
- **Voltage**: Voltage level during the electricity consumption period.
- **Global_intensity**: Total current intensity consumed by the household.
- **Sub_metering_1**: Electricity consumption in sub-metering 1 (e.g., kitchen).
- **Sub_metering_2**: Electricity consumption in sub-metering 2 (e.g., laundry).
- **Sub_metering_3**: Electricity consumption in sub-metering 3 (e.g., water heater).

## Project Structure

```
├── data/                                # Directory for storing dataset files
│   ├── household_power_consumption.txt   # Original dataset
│   ├── preprocessed_household_power_consumption.csv   # Preprocessed dataset
│   └── feature_engineered_household_power_consumption.csv  # Feature engineered dataset
├── notebooks/                           # Jupyter Notebooks for different stages
│   ├── EDA_and_Preprocessing.ipynb      # Exploratory Data Analysis and Preprocessing
│   ├── ARIMA_Model.ipynb                # ARIMA Model for time series forecasting
│   ├── SARIMA_Model.ipynb               # SARIMA Model for time series forecasting
│   ├── LSTM_Model.ipynb                 # LSTM Model for time series forecasting
│   └── Model_Tuning.ipynb               # Grid search for model parameter tuning
├── results/                             # Directory for storing results
│   └── plots/                           # Directory for storing visualization plots
│       ├── arima_forecast.png           # ARIMA model forecast plot
│       ├── sarima_forecast.png          # SARIMA model forecast plot
│       ├── lstm_forecast.png            # LSTM model forecast plot
│       ├── future_forecast.png          # Future forecasting plot
│       └── global_active_power_over_time.png  # EDA plot
├── scripts/                             # Directory for main scripts
│   ├── preprocess_data.py               # Script for data preprocessing
│   ├── exploratory_data_analysis.py     # Script for EDA and visualization
│   ├── feature_engineering.py           # Script for feature engineering
│   ├── arima_forecasting.py             # Script for ARIMA model forecasting
│   ├── sarima_forecasting.py            # Script for SARIMA model forecasting
│   ├── lstm_forecasting.py              # Script for LSTM model forecasting
│   └── model_tuning.py                  # Script for model parameter tuning
├── requirements.txt                     # Python dependencies
├── README.md                            # Project overview and instructions
└── LICENSE                              # License information
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/shahram8708/household-electricity-consumption.git
   cd household-electricity-consumption
   ```

2. **Install the required Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This installs necessary libraries such as NumPy, pandas, Matplotlib, scikit-learn, TensorFlow, and others required for data manipulation, modeling, and visualization.

## Usage

1. **Exploratory Data Analysis and Preprocessing**:
   - Open and run `EDA_and_Preprocessing.ipynb` in Jupyter Notebook to explore data patterns, visualize relationships between features, and preprocess the dataset.

2. **Modeling and Forecasting**:
   - Explore different models:
     - Use `ARIMA_Model.ipynb` for ARIMA model forecasting.
     - Use `SARIMA_Model.ipynb` for SARIMA model forecasting.
     - Use `LSTM_Model.ipynb` for LSTM model forecasting.
   
3. **Model Tuning**:
   - Explore parameter tuning using `Model_Tuning.ipynb` to optimize model performance.

4. **Visualization**:
   - Check the `results/plots` directory for visualizations generated during EDA and forecasting.

5. **Interpreting Results**:
   - Review model forecasts and evaluation metrics to interpret results and identify consumption patterns.

## Contributions

Contributions are welcome! If you have suggestions for improvements, please:

- Fork the repository.
- Create a new branch (`git checkout -b feature-improvements`).
- Make modifications and commit changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature-improvements`).
- Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
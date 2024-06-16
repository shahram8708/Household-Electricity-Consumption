
import pandas as pd
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA

def tune_arima_model(data):
    train_data, _ = train_test_split(data['Global_active_power'], test_size=0.2, shuffle=False)

    p = range(0, 6)
    d = range(0, 2)
    q = range(0, 6)
    pdq = [(x, y, z) for x in p for y in d for z in q]

    grid_search = GridSearchCV(ARIMA(), param_grid={'order': pdq}, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(train_data)

    best_params = grid_search.best_params_
    print(f'Best ARIMA parameters: {best_params}')

if __name__ == "__main__":
    data = pd.read_csv('../data/feature_engineered_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    tune_arima_model(data)

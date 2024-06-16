
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Global_active_power'], label='Global Active Power')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.title('Global Active Power over Time')
    plt.legend()
    plt.savefig('../results/plots/global_active_power_over_time.png')
    plt.show()

    data.hist(bins=50, figsize=(20,15))
    plt.savefig('../results/plots/histograms.png')
    plt.show()

    sns.pairplot(data)
    plt.savefig('../results/plots/pairplot.png')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('../data/preprocessed_household_power_consumption.csv', index_col='datetime', parse_dates=True)
    exploratory_data_analysis(data)

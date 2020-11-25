# This is data cleaning and processing script based on the EDA notebook.

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing

# Read in the raw, unprocesed data
data = pd.read_csv('data/raw/owid-covid-data.csv')

# Drop all rows where date is not equal to 01.10.2020
data.drop(data[data['date'] != '2020-10-01'].index, inplace=True)

# Drop rows (data instances) that are not single country instances
data.drop(data[data['location'] == 'World'].index, inplace=True)
data.drop(data[data['location'] == 'International'].index, inplace=True)

# Drop rows not considered to be important for the analysis. See the related EDA notebook for explanation.
data.drop(
    ['aged_65_older', 'continent', 'date', 'new_cases', 'new_cases_per_million', 'new_cases_smoothed',
     'new_cases_smoothed_per_million', 'new_deaths', 'new_deaths_per_million', 'new_deaths_smoothed',
     'new_deaths_smoothed_per_million', 'new_tests', 'new_tests_per_thousand', 'new_tests_smoothed',
     'new_tests_smoothed_per_thousand', 'tests_units', 'total_cases', 'total_deaths', 'total_tests', 'tests_per_case',
     'positive_rate', 'population', 'total_tests_per_thousand', 'handwashing_facilities', 'extreme_poverty'],
    axis=1, inplace=True)


# Save locations and clean data for later refrence. We use location and iso_code for visualization.
data['location'].to_csv('data/raw/locations.csv', index=False)
data.to_csv('data/clean/clean.csv', index=False)

# Drop location and iso_code as these are unique values and does not offer anything to the clustering.
data.drop(['location', 'iso_code'], axis=1, inplace=True)

# Implement imputer, fit model to data, and impute missing values
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(data)
imputed_df = imp_mean.transform(data)
data = pd.DataFrame(imputed_df, columns=data.columns)

# Scale data using normalization to better comply with distance based clustering methods
X = data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
data = pd.DataFrame(x_scaled, columns=data.columns)


# Create new re-weighted dataset where COVID-19 features are weighted higher
data_weighted = data.copy()

data_weighted[['total_cases_per_million', 'total_deaths_per_million', 'stringency_index']
              ] = data_weighted[['total_cases_per_million', 'total_deaths_per_million', 'stringency_index']] * 2


# Save processed data to file
data.to_csv('data/processed/processed.csv', index=False)
data_weighted.to_csv('data/processed/processed_weighted.csv', index=False)

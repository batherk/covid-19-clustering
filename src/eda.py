## THIS FILE cleans data according to the step-wise process illustrated in the notebook

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing

# Read in the raw, unprocesed data
data = pd.read_csv('../data/raw/owid-covid-data.csv')

# Drop all rows where date is not equal to 01.10.2020
data.drop(data[data['date'] != "2020-10-01"].index, inplace = True)

# Drop rows (data instances) that are not single country instances
data.drop(data[data['location'] == "World"].index, inplace = True) 
data.drop(data[data['location'] == "International"].index, inplace = True)

# Drop rows not considered to be important for the analysis. The EDA-notebook for explanation. 
data.drop(['aged_65_older', 'continent', 'date', 'iso_code', 'new_cases', 'new_cases_per_million', 
'new_cases_smoothed', 'new_cases_smoothed_per_million', 'new_deaths', 'new_deaths_per_million', 
'new_deaths_smoothed', 'new_deaths_smoothed_per_million', 'new_tests', 'new_tests_per_thousand', 
'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units', 'total_cases', 'total_deaths', 
'total_tests', 'tests_per_case', 'positive_rate'], axis=1, inplace=True)


# Save locations and clean data for later refrence. We use location for indexing of the clusters.
data['location'].to_csv("../data/raw/locations.csv", index=False)
data.to_csv("../data/clean/clean.csv", index=False)

# Drop lacation. These are unique values and does not offer anything to the clustering. We save the feature for future indexing of clusters.
data.drop(['location'], axis=1, inplace=True)

# Implement imputer, fit model to data, and impute missing values
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(data)

imputed_df = imp_mean.transform(data)
data = pd.DataFrame(imputed_df, columns=data.columns)

# Scale data using normalization to better comply with distance based clustering methods
x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled, columns=data.columns)


# Create new re-weighted dataset where covid-features are weighted higher
data_reweighted = data

data_reweighted[['total_cases_per_million', 'total_deaths_per_million', 'total_tests_per_thousand','stringency_index']] = data_reweighted[['total_cases_per_million', 'total_deaths_per_million', 'total_tests_per_thousand','stringency_index']]*2


# Save processed data to file
data.to_csv("../data/processed/processed.csv", index=False)
data.to_csv("../data/processed/processed_reweighted.csv", index=False)




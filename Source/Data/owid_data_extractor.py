import numpy as np
import csv

ATTRIBUTES = ['iso_code', 'continent', 'location', 'total_cases_per_million', 'total_deaths_per_million', 'tests_per_case', 'positive_rate', 'tests_units', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']

def extract_data_from_owid_covid_data():
    
    line_count = 0
    attributes_indexes = []
    data = []

    csv_file = open('../../Data/Raw/owid-covid-data.csv')
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    for row in csv_reader:
        if line_count == 0:
            for attribute in ATTRIBUTES:
                attributes_indexes.append(row.index(attribute))
        elif row[3] == "2020-10-01":
            country_data = []
            for i in attributes_indexes:
                 country_data.append(row[i])
            data.append(country_data)
        line_count += 1
    csv_file.close()

    return np.array(data)

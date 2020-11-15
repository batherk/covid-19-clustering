# Which problem are you addressing?

The coronavirus disease has turned the world upside down and continues to wreak
havoc into late 2020. Different countries faced the pandemic with various starting
points when it comes to public health, healthcare spending, life expectancy, etc.
We want to use clustering to look for hidden patterns in aggregated COVID-19 data
for different countries.

# Which dataset do you plan to use in your projects?

We plan to use the "Mortality risk of COVID-19"-dataset from Our World in Data
[https://ourworldindata.org/mortality-risk-covid](https://ourworldindata.org/mortality-risk-covid). The dataset contains
country-by-country data on mortality risk of the COVID-19 pandemic.
We will also look into adding new features to this dataset. There is demographic
information available that might prove useful for the predictors. For instance we
found the data set of the share of people who say they trust the national government
very interesting: [https://ourworldindata.org/grapher/share-who-trust-government](https://ourworldindata.org/grapher/share-who-trust-government).
Even though this data set does not contain all the countries in the world, we will add
it to the main data set.

# Which machine learning methods do you plan to apply?

We want to use the following clustering methods:

- k-means
- Mean-shift
- Birch
- Hierarchical clustering
- Spectral clustering

# What kind of experiments do you plan to run?

We will start off by doing an exploratory data analysis (EDA) of our dataset. The aim
here is to detect missing values, duplicate features, and outliers. We will then
explore individual features and groups of features. Based on the EDA we will do a
data cleanup (imputing, standardizing, drop features, one-hot-encoding etc). The
reporting of COVID-cases has varied substantially between countries, so EDA will be
important ahead of any prediction.

After the cleanup we will continue with our prediction. The aim here is to run our
dataset through the clustering algorithms mentioned above. As part of our
interpretation we will compare the results of the method against COVID-deaths per
million and COVID-cases per million. We also plan to try different implementations of
the methods, e.g. initializing k-means with k-means++ and random initialization.

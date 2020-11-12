import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Read the processed data from our EDA
X = pd.read_csv('data/processed/processed.csv')

# Initialize KMeans
k_means = KMeans(n_clusters=20)

# Fit data to model and predict labels for data
k_means.fit_predict(X)

# Get the predicted labels
y_pred = k_means.labels_

print(y_pred)

joblib_filename = 'models/k_means.joblib'
joblib.dump(k_means, joblib_filename)

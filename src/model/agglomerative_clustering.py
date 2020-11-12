import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import joblib

# Read the processed data from our EDA
X = pd.read_csv('data/processed/processed.csv')

# Initialize Hierarchical clustering with Agglomerative clustering
agglomerative_clustering = AgglomerativeClustering(n_clusters=20)

# Fit data to model and predict labels for data
agglomerative_clustering.fit_predict(X)

# Get the predicted labels
y_pred = agglomerative_clustering.labels_

print(y_pred)

joblib_filename = 'models/agglomerative_clustering.joblib'
joblib.dump(agglomerative_clustering, joblib_filename)

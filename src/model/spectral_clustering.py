import pandas as pd
from sklearn.cluster import SpectralClustering
import joblib

# Read the processed data from our EDA
X = pd.read_csv('data/processed/processed.csv')

# Initialize Spectral Clusering model
spectral_clustering = SpectralClustering(
    n_clusters=20,
    affinity='nearest_neighbors',
    n_init=100,
    assign_labels='discretize'
)

# Fit data to model and predict labels for data
spectral_clustering.fit_predict(X)

# Get the predicted labels (clusters) for each row of our dataset
y_pred = spectral_clustering.labels_

print(y_pred)

joblib_filename = 'models/spectral_clustering.joblib'
joblib.dump(spectral_clustering, joblib_filename)

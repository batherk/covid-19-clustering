# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import KMeans

# Initialize KMeans
k_means = KMeans()

model = dict({
    'model': k_means,
    'metadata': {
        'name': 'k-Means',
        'abbreviation': 'KM',
        'datetime': str(datetime.now()),
        'hyperparameters': {}
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
k_means.fit_predict(X)
labels = k_means.labels_
save_path = 'results/k_means/default_parameters'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/k_means/default_parameters.joblib'
joblib.dump(model, joblib_filename)

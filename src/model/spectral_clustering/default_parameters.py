# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import SpectralClustering

# Initialize spectral clusering
spectral_clustering = SpectralClustering()

model = dict({
    'model': spectral_clustering,
    'metadata': {
        'name': 'Spectral Clusering',
        'abbreviation': 'SC',
        'datetime': str(datetime.now()),
        'hyperparameters': {},
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
spectral_clustering.fit_predict(X)
labels = spectral_clustering.labels_
save_path = 'results/spectral_clustering/default_parameters'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/spectral_clustering/default_parameters.joblib'
joblib.dump(model, joblib_filename)

# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import SpectralClustering

# Hyperparameters
n_clusters = 20
n_init = 100
affinity = 'rbf'
assign_labels = 'discretize'

# Initialize spectral clusering
spectral_clustering = SpectralClustering(
    n_clusters=n_clusters,
    n_init=n_init,
    affinity=affinity,
    assign_labels=assign_labels
)

model = dict({
    'model': spectral_clustering,
    'metadata': {
        'name': 'Spectral Clusering',
        'abbreviation': 'SC',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': n_clusters,
            'n_init': n_init,
            'affinity': affinity,
            'assign_labels': assign_labels
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
spectral_clustering.fit_predict(X)
labels = spectral_clustering.labels_
save_path = 'results/spectral_clustering/optimized'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/spectral_clustering/optimized.joblib'
joblib.dump(model, joblib_filename)

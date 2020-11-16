# type: ignore
import pathmagic  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clustering_metrics
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
pd.DataFrame(labels, columns=['cluster']).to_csv(
    'results/spectral_clustering/raw.csv', index=False)

save_clustering_metrics(X, labels, 'results/spectral_clustering')

# Persist model and metadata
joblib_filename = 'models/spectral_clustering.joblib'
joblib.dump(model, joblib_filename)

# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import DBSCAN

# Hyperparameters
eps = 0.95
min_samples = 2

# Initialize DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

model = dict({
    'model': dbscan,
    'metadata': {
        'name': 'DBSCAN',
        'abbreviation': 'DBS',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'eps': eps,
            'min_samples': min_samples
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
dbscan.fit_predict(X)
labels = dbscan.labels_
save_path = 'results/dbscan/optimized'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/dbscan/optimized.joblib'
joblib.dump(model, joblib_filename)

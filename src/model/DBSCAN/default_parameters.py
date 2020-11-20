# type: ignore
import pathsetup  # noqa
import pandas as pd
from sklearn.cluster import DBSCAN
import joblib
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from datetime import datetime

# Initialize DBSCAN
dbscan = DBSCAN()

model = dict({
    'model': dbscan,
    'metadata': {
        'name': 'DBSCAN',
        'abbreviation': 'DBS',
        'datetime': str(datetime.now()),
        'hyperparameters': {},
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
dbscan.fit_predict(X)
labels = dbscan.labels_
save_path = 'results/dbscan/default_parameters'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/dbscan/default_parameters.joblib'
joblib.dump(model, joblib_filename)

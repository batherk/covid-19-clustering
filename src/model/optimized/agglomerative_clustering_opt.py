# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import AgglomerativeClustering

# Hyperparameters
affinity = 'l1'
linkage = 'complete'
distance_threshold = 7.9


# Initialize hierarchical clustering with agglomerative clustering
agglomerative_clustering = AgglomerativeClustering(n_clusters=None,
                                                   affinity=affinity, linkage=linkage, distance_threshold=distance_threshold)

model = dict({
    'model': agglomerative_clustering,
    'metadata': {
        'name': 'Agglomerative Clustering',
        'abbreviation': 'AC',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'affinity': affinity,
            'linkage': linkage,
            'distance_threshold': distance_threshold
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
agglomerative_clustering.fit_predict(X)
labels = agglomerative_clustering.labels_
save_path = 'results/agglomerative_clustering/optimized'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/agglomerative_clustering_opt.joblib'
joblib.dump(model, joblib_filename)

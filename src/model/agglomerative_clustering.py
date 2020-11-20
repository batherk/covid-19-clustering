# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import AgglomerativeClustering

# Hyperparameters
n_clusters = 20

# Initialize hierarchical clustering with agglomerative clustering
agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters)

model = dict({
    'model': agglomerative_clustering,
    'metadata': {
        'name': 'Agglomerative Clustering',
        'abbreviation': 'AC',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': n_clusters
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
agglomerative_clustering.fit_predict(X)
labels = agglomerative_clustering.labels_
save_clusters_as_csv(labels, 'results/agglomerative_clustering')
save_clustering_metrics_as_csv(X, labels, 'results/agglomerative_clustering')

# Persist model and metadata
joblib_filename = 'models/agglomerative_clustering.joblib'
joblib.dump(model, joblib_filename)

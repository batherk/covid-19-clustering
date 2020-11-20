# type: ignore
import pathsetup  # noqa
from sklearn.cluster import AgglomerativeClustering
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from datetime import datetime
import pandas as pd
import joblib


# Initialize hierarchical clustering with agglomerative clustering
agglomerative_clustering = AgglomerativeClustering()

model = dict({
    'model': agglomerative_clustering,
    'metadata': {
        'name': 'Agglomerative Clustering',
        'abbreviation': 'AC',
        'datetime': str(datetime.now()),
        'hyperparameters': {}
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
agglomerative_clustering.fit_predict(X)
labels = agglomerative_clustering.labels_
save_path = 'results/agglomerative_clustering/default_parameters'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/agglomerative_clustering/default_parameters.joblib'
joblib.dump(model, joblib_filename)

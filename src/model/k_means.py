# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import KMeans

# Hyperparameters
k = 20

# Initialize KMeans
k_means = KMeans(n_clusters=k)

model = dict({
    'model': k_means,
    'metadata': {
        'name': 'k-Means',
        'abbreviation': 'KMeans',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': k,
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
k_means.fit_predict(X)
labels = k_means.labels_
save_clusters_as_csv(labels, 'results/k_means')
save_clustering_metrics_as_csv(X, labels, 'results/k_means')

# Persist model and metadata
joblib_filename = 'models/k_means.joblib'
joblib.dump(model, joblib_filename)

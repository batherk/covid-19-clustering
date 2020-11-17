# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clustering_metrics
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
pd.DataFrame(labels, columns=['cluster']).to_csv(
    'results/k_means/raw.csv', index=False)

save_clustering_metrics(X, labels, 'results/k_means')

# Persist model and metadata
joblib_filename = 'models/k_means.joblib'
joblib.dump(model, joblib_filename)

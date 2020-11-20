# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clustering_metrics_as_csv
from sklearn.cluster import Birch


# Hyperparameters
n_clusters = 2
threshold = 0.6
branching_factor = 10

# Initialize BIRCH model
birch = Birch(n_clusters=n_clusters)

model = dict({
    'model': birch,
    'metadata': {
        'name': 'BIRCH',
        'abbreviation': 'birch',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': n_clusters,
            'threshold': threshold,
            'branching_factor': branching_factor
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
birch.fit_predict(X)
labels = birch.labels_
pd.DataFrame(labels, columns=['cluster']).to_csv(
    'results/birch/clusters.csv', index=False)
save_clustering_metrics_as_csv(X, labels, 'results/birch')

# Persist model and metadata
joblib_filename = 'models/birch.joblib'
joblib.dump(model, joblib_filename)

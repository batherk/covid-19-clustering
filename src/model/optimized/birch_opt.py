# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import Birch


# Hyperparameters
threshold = 2
branching_factor = 1

# Initialize BIRCH model
birch = Birch(n_clusters=None, threshold=threshold,
              branching_factor=branching_factor)

model = dict({
    'model': birch,
    'metadata': {
        'name': 'BIRCH',
        'abbreviation': 'BI',
        'datetime': str(datetime.now()),
        'hyperparameters': {
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
save_path = 'results/birch/optimized'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Train model
joblib_filename = 'models/birch/optimized.joblib'
joblib.dump(model, joblib_filename)

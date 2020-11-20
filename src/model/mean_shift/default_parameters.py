# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import MeanShift


# Initialize spectral clusering
mean_shift = MeanShift()

model = dict({
    'model': mean_shift,
    'metadata': {
        'name': 'MeanShift',
        'abbreviation': 'MS',
        'datetime': str(datetime.now()),
        'hyperparameters': {},
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
mean_shift.fit_predict(X)
labels = mean_shift.labels_
save_path = 'results/mean_shift/default_parameters'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/mean_shift/default_parameters.joblib'
joblib.dump(model, joblib_filename)

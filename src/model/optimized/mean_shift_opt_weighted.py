# type: ignore
import pathsetup  # noqa
import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clusters_as_csv, save_clustering_metrics_as_csv
from sklearn.cluster import MeanShift


# Hyperparameters
bandwidth = 0.40

# Initialize mean shift
mean_shift = MeanShift(bandwidth=bandwidth)

model = dict({
    'model': mean_shift,
    'metadata': {
        'name': 'MeanShift',
        'abbreviation': 'MS',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'bandwidth': bandwidth
        },
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed_weighted.csv')

# Cluster and save results
mean_shift.fit_predict(X)
labels = mean_shift.labels_
save_path = 'results/mean_shift/optimized_weighted'
save_clusters_as_csv(labels, save_path)
save_clustering_metrics_as_csv(X, labels, save_path)

# Persist model and metadata
joblib_filename = 'models/mean_shift_opt_weighted.joblib'
joblib.dump(model, joblib_filename)

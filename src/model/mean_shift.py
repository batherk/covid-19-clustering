import joblib
import pandas as pd
from datetime import datetime
from src.utils import save_clustering_metrics
from sklearn.cluster import MeanShift

# Hyperparameters


# Initialize mean shift
mean_shift = MeanShift()

model = dict({
    'model': mean_shift,
    'metadata': {
        'name': 'Mean Shift',
        'abbreviation': 'MS',
        'datetime': str(datetime.now()),
        'hyperparameters': {
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
mean_shift.fit_predict(X)
labels = mean_shift.labels_
pd.DataFrame(labels, columns=['cluster']).to_csv(
    'results/mean_shift/raw.csv', index=False)

save_clustering_metrics(X, labels, 'results/mean_shift')

# Persist model and metadata
joblib_filename = 'models/mean_shift.joblib'
joblib.dump(model, joblib_filename)

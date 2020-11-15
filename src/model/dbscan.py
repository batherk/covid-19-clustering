import pandas as pd
from sklearn.cluster import DBSCAN
import joblib
from datetime import datetime

# Hyperparameters
eps = 0.95
min_samples = 2

# Initialize DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

model = dict({
    'model': dbscan,
    'metadata': {
        'name': 'DBSCAN',
        'abbreviation': 'dbscan',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'eps': eps,
            'min_samples': min_samples
        },
        'metrics': {
            'silhouette coefficient': 0
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Train model
dbscan.fit_predict(X)
y_pred = dbscan.labels_
print(y_pred)

# Persist model and metadata
joblib_filename = 'models/dbscan.joblib'
joblib.dump(model, joblib_filename)

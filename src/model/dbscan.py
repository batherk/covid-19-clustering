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
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
dbscan.fit_predict(X)
labels = dbscan.labels_
pd.DataFrame(labels, columns=['cluster']).to_csv(
    'results/dbscan/raw.csv', index=False)

# Persist model and metadata
joblib_filename = 'models/dbscan.joblib'
joblib.dump(model, joblib_filename)

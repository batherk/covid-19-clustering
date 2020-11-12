import pandas as pd
from sklearn.cluster import Birch
import joblib
from datetime import datetime

# Hyperparameters
n_clusters = 20

# Initialize BIRCH model
birch = Birch(n_clusters=n_clusters)

model = dict({
    'model': birch,
    'metadata': {
        'name': 'BIRCH',
        'abbreviation': 'birch',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': n_clusters
        },
        'metrics': {
            'silhouette coefficient': 0
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Train model
birch.fit_predict(X)
y_pred = birch.labels_
print(y_pred)

# Train model
joblib_filename = 'models/birch.joblib'
joblib.dump(model, joblib_filename)

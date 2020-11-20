import pandas as pd
from sklearn.cluster import DBSCAN
import joblib
from datetime import datetime

# Initialize DBSCAN
dbscan = DBSCAN()

model = dict({
    'model': dbscan,
    'metadata': {
        'name': 'DBSCAN',
        'abbreviation': 'DBS',
        'datetime': str(datetime.now()),
        'hyperparameters': {},
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Train model
dbscan.fit_predict(X)
y_pred = dbscan.labels_
print(y_pred)

# Persist model and metadata
joblib_filename = 'models/dbscan_default_parameters.joblib'
joblib.dump(model, joblib_filename)

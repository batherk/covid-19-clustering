import pandas as pd
from sklearn.cluster import MeanShift
import joblib
from datetime import datetime


# Initialize spectral clusering
mean_shift = MeanShift()

model = dict({
    'model': mean_shift,
    'metadata': {
        'name': 'MeanShift',
        'abbreviation': 'MS',
        'datetime': str(datetime.now()),
        'hyperparameters': {},
        'metrics': {
            'silhouette coefficient': 0
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Train model
mean_shift.fit_predict(X)
y_pred = mean_shift.labels_
print(y_pred)

# Persist model and metadata
joblib_filename = 'models/mean_shift/default_parameters.joblib'
joblib.dump(model, joblib_filename)

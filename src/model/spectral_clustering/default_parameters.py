import pandas as pd
from sklearn.cluster import SpectralClustering
import joblib
from datetime import datetime


# Initialize spectral clusering
spectral_clustering = SpectralClustering()

model = dict({
    'model': spectral_clustering,
    'metadata': {
        'name': 'Spectral Clusering',
        'abbreviation': 'SC',
        'datetime': str(datetime.now()),
        'hyperparameters': {},
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Train model
spectral_clustering.fit_predict(X)
y_pred = spectral_clustering.labels_
print(y_pred)

# Persist model and metadata
joblib_filename = 'models/spectral_clustering/default_parameters.joblib'
joblib.dump(model, joblib_filename)

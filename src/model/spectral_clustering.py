import pandas as pd
from sklearn.cluster import SpectralClustering
import joblib
from datetime import datetime

# Hyperparameters
n_clusters = 20
n_init = 100
affinity = 'rbf'
assign_labels = 'discretize'

# Initialize spectral clusering
spectral_clustering = SpectralClustering(
    n_clusters=n_clusters,
    n_init=n_init,
    affinity=affinity,
    assign_labels=assign_labels
)

model = dict({
    'model': spectral_clustering,
    'metadata': {
        'name': 'Spectral Clusering',
        'abbreviation': 'SC',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': n_clusters,
            'n_init': n_init,
            'affinity': affinity,
            'assign_labels': assign_labels
        },
        'metrics': {
            'silhouette coefficient': 0
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Train model
spectral_clustering.fit_predict(X)
y_pred = spectral_clustering.labels_
print(y_pred)

# Persist model and metadata
joblib_filename = 'models/spectral_clustering.joblib'
joblib.dump(model, joblib_filename)

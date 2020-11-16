import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import joblib
from datetime import datetime

# Hyperparameters
n_clusters = 20

# Initialize hierarchical clustering with agglomerative clustering
agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters)

model = dict({
    'model': agglomerative_clustering,
    'metadata': {
        'name': 'Agglomerative Clustering',
        'abbreviation': 'AC',
        'datetime': str(datetime.now()),
        'hyperparameters': {
            'n_clusters': n_clusters
        }
    }
})

# Read the processed data from the EDA
X = pd.read_csv('data/processed/processed.csv')

# Cluster and save results
agglomerative_clustering.fit_predict(X)
labels = agglomerative_clustering.labels_
pd.DataFrame(labels, columns=['cluster']).to_csv(
    'results/agglomerative_clustering/raw.csv', index=False)

# Persist model and metadata
joblib_filename = 'models/agglomerative_clustering.joblib'
joblib.dump(model, joblib_filename)

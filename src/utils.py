import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def mapIndexToLocation(index):
    return pd.read_csv('data/raw/locations.csv')['location'].get(index)


def save_clusters_as_csv(labels, path):
    pd.DataFrame(labels, columns=['cluster']).to_csv(
        f'{path}/clusters.csv', index=False)


def save_clustering_metrics_as_csv(X, labels, path):
    clustering_metrics = pd.DataFrame({
        'metric_name': [
            'Silhouette Coefficient',
            'Davies-Bouldin Index',
            'Calinski-Harabasz Index'
        ],
        'value': [
            silhouette_score(X, labels),
            davies_bouldin_score(X, labels),
            calinski_harabasz_score(X, labels)
        ],
    })
    clustering_metrics.to_csv(f'{path}/clustering_metrics.csv', index=False)

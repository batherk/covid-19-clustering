import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt


def mapIndexToLocation(index):
    return pd.read_csv('data/raw/locations.csv')['location'].get(index)


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


def save_clustering_metrics_as_pdf(X, labels, path):
    silhouette_coefficient = silhouette_score(X, labels)
    davies_bouldin_index = davies_bouldin_score(X, labels)
    calinski_harabasz_index = calinski_harabasz_score(X, labels)

    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.text(0, 15, 'Clustering metrics', fontsize=60)
    ax.text(
        0, 10, f'Silhouette coefficient: {silhouette_coefficient:.4f}', fontsize=36)
    ax.text(
        0, 7, f'Davies-Bouldin index: {davies_bouldin_index:.4f}', fontsize=36)
    ax.text(
        0, 4, f'Calinski-Harabasz index: {calinski_harabasz_index:.4f}', fontsize=36)
    plt.savefig(f'{path}/clustering_metrics.pdf')

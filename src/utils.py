import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import csv


def mapIndexToLocation(index):
    return pd.read_csv('data/raw/locations.csv')['location'].get(index)


def save_clustering_metrics(X, labels, path):
    silhouette_coefficient = silhouette_score(X, labels)
    davies_bouldin_index = davies_bouldin_score(X, labels)
    calinski_harabasz_index = calinski_harabasz_score(X, labels)

    with open(f'{path}/clustering_metrics.csv', mode='w', newline='') as clustering_metrics:
        csv_writer = csv.writer(clustering_metrics, delimiter=',')
        csv_writer.writerow(
            ['Silhouette Score', 'Davies Bouldin Score', 'Calinski-Harabasz Score'])
        csv_writer.writerow(
            [silhouette_coefficient, davies_bouldin_index, calinski_harabasz_index])

import pandas as pd
from sklearn.cluster import DBSCAN
import joblib
import seaborn as sns
sns.set()
# import numpy as np
# from sklearn.metrics import silhouette_score
# from sklearn.neighbors import NearestNeighbors
# from matplotlib import pyplot as plt

# Read the processed data from our EDA
X = pd.read_csv('data/processed/processed.csv')

# Find the optimum epsilon distance (eps)
# from https://medium.com/@mohantysandip/a-step-by-step-approach-to-solve-dbscan-algorithms-by-tuning-its-hyper-parameters-93e693a91289

# Create elbow plot using kNN
# kNN = NearestNeighbors(n_neighbors=2)
# neighbors = kNN.fit(X)
# distances, indices = neighbors.kneighbors(X)
# distances = np.sort(distances, axis=0)[::-1]
# distances = distances[:, 1]
# plt.plot(distances)
# plt.savefig('neighbors_plot.pdf')

# epsilon_range = np.arange(0.1, 1, 0.01)
# silhouette_averages = []

# for epsilon in epsilon_range:
#     dbscan = DBSCAN(eps=epsilon, min_samples=5).fit(X)
#     core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
#     core_samples_mask[dbscan.core_sample_indices_] = True
#     labels = dbscan.labels_
#     if len(set(labels)) >= 2:
#         silhouette_average = silhouette_score(X, labels)
#         silhouette_averages.append((epsilon, silhouette_average))
#         print(
#             f'epsilon={epsilon:.2f}: average. silhouette score={silhouette_average:>7.4f}')

# plt.clf()
# plt.plot([eps for eps, _ in silhouette_averages],
#          [sa for _, sa in silhouette_averages])
# plt.savefig('eps_avg_silhouette_score.pdf')

# optimum_epsilon = max(silhouette_averages, key=lambda x: x[1])[0]
# print(optimum_epsilon)

# Initialize DBSCAN model
dbscan = DBSCAN(eps=0.9, min_samples=5).fit(X)

# Fit data to model and predict labels for data
dbscan.fit_predict(X)

# Get the predicted labels (clusters) for each row of our dataset
y_pred = dbscan.labels_

print(y_pred)

joblib_filename = 'models/dbscan.joblib'
joblib.dump(dbscan, joblib_filename)

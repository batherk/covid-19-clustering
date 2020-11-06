import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix

# Read the processed data from our EDA
data = pd.read_csv('../../Data/Processed/processed.csv')

# Initialize Hierarchical clustering with Agglomerative clustering
k_means = KMeans(n_clusters=20)

# Fit data to model and predict labels for data
k_means.fit_predict(data)

# Get the predicted labels
y_pred = k_means.labels_

print(y_pred)
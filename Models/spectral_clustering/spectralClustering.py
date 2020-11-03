import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering

# Read the processed data from our EDA
data = pd.read_csv('../../Data/Processed/processed.csv')

# Initialize Spectral Clusering model
sc = SpectralClustering(n_clusters=20, affinity='nearest_neighbors', n_init=100, assign_labels='discretize')

# Fit data to model and predict labels for data 
sc.fit_predict(data)

# Get the predicted labels (clusters) for each row of our dataset 
y_pred = sc.labels_.astype(np.int)

print(y_pred)






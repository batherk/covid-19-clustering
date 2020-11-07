import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import contingency_matrix

# Read the processed data from our EDA
data = pd.read_csv('../Data/Processed/processed.csv')

# Initialize DBSCAN model
dbscan = DBSCAN(eps=2, min_samples=2)

# Fit data to model and predict labels for data
dbscan.fit_predict(data)

# Get the predicted labels (clusters) for each row of our dataset
y_pred = dbscan.labels_.astype(np.int)
cm = contingency_matrix(data.index, y_pred)

print(y_pred)
print(cm)

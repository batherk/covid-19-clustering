import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Read the processed data from our EDA
data = pd.read_csv('../Data/Processed/processed.csv')

# Initialize Hierarchical clustering with Agglomerative clustering
ac = AgglomerativeClustering(n_clusters=20)

# Fit data to model and predict labels for data
ac.fit_predict(data)

# Get the predicted labels
y_pred = ac.labels_

print(y_pred)

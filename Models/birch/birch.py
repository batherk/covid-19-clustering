import pandas as pd
import numpy as np
from sklearn.cluster import Birch

# Read the processed data from our EDA
data = pd.read_csv('../../Data/Processed/processed.csv')

# Initialize Birch model
birch = Birch(n_clusters=20)

# Fit data to model and predict labels for data
birch.fit_predict(data)

# Get the predicted labels (clusters) for each row of our dataset
y_pred = birch.labels_.astype(np.int)

print(y_pred)

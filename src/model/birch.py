import pandas as pd
from sklearn.cluster import Birch
import joblib

# Read the processed data from our EDA
X = pd.read_csv('data/processed/processed.csv')

# Initialize Birch model
birch = Birch(n_clusters=20)

# Fit data to model and predict labels for data
birch.fit_predict(X)

# Get the predicted labels (clusters) for each row of our dataset
y_pred = birch.labels_

print(y_pred)

joblib_filename = 'models/birch.joblib'
joblib.dump(birch, joblib_filename)

import joblib
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

data = pd.read_csv('data/processed/processed.csv')


model_file_names = [
    "agglomerative_clustering/default_parameters.joblib",
    "birch/default_parameters.joblib",
    "dbscan/default_parameters.joblib",
    "k_means/default_parameters.joblib",
    "spectral_clustering/default_parameters.joblib",
    "mean_shift/default_parameters.joblib"
]

models = []

for model_file_name in model_file_names:
    path = "models/" + model_file_name
    models.append(joblib.load(path))


scores = []
iterations_per_method = 100

for index, model in enumerate(models):
    for iteration in range(iterations_per_method):
        prediction = model['model'].fit_predict(X=data)
        sc = silhouette_score(data, prediction)
        db = davies_bouldin_score(data, prediction)
        ch = calinski_harabasz_score(data, prediction)
        scores.append((model, iteration, sc, db, ch))

joblib.dump(
    scores, "results/evaluation/all_methods_with_default_parameters/scores.joblib")

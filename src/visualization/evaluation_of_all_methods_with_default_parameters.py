import joblib
import matplotlib.pyplot as plt
import numpy as np

scores = joblib.load(
    "results/evaluation/all_methods_with_default_parameters/scores.joblib")
methods, iteration_numbers, silhouette_scores_all, davies_bouldin_scores_all, calinski_harabasz_scores_all = zip(
    *scores)

method_abbreviations = list(
    set([method["metadata"]["abbreviation"] for method in methods]))
method_abbreviations.sort()

methods_x_axis = [i for i in range(len(method_abbreviations))]


silhouette_scores_by_method = [[]for i in range(len(method_abbreviations))]
davies_bouldin_scores_by_method = [[]for i in range(len(method_abbreviations))]
calinski_harabasz_scores_by_method = [[]
                                      for i in range(len(method_abbreviations))]

for i, method in enumerate(methods):
    method_id = method_abbreviations.index(method["metadata"]["abbreviation"])
    silhouette_scores_by_method[method_id].append(silhouette_scores_all[i])
    davies_bouldin_scores_by_method[method_id].append(
        davies_bouldin_scores_all[i])
    calinski_harabasz_scores_by_method[method_id].append(
        calinski_harabasz_scores_all[i])


silhouette_score_avg_per_method = [
    sum(score_list) / len(score_list) for score_list in silhouette_scores_by_method]
davies_bouldin_score_avg_per_method = [sum(
    score_list) / len(score_list) for score_list in davies_bouldin_scores_by_method]
calinski_harabasz_score_avg_per_method = [sum(
    score_list) / len(score_list) for score_list in calinski_harabasz_scores_by_method]

silhouette_score_std_per_method = [
    np.std(score_list) for score_list in silhouette_scores_by_method]
davies_bouldin_score_std_per_method = [
    np.std(score_list) for score_list in davies_bouldin_scores_by_method]
calinski_harabasz_score_std_per_method = [
    np.std(score_list) for score_list in calinski_harabasz_scores_by_method]


plt.bar(methods_x_axis, silhouette_score_avg_per_method,
        yerr=silhouette_score_std_per_method, color="blue")
plt.xlabel('Clustering methods')
plt.ylabel('Silhouette score')
plt.xticks(methods_x_axis, method_abbreviations)
plt.savefig(
    "results/evaluation/all_methods_with_default_parameters/silhouette_scores")

plt.bar(methods_x_axis, davies_bouldin_score_avg_per_method,
        yerr=davies_bouldin_score_std_per_method, color="blue")
plt.ylabel('Davies-Bouldin score')
plt.savefig(
    "results/evaluation/all_methods_with_default_parameters/davies_bouldin_score")

plt.bar(methods_x_axis, calinski_harabasz_score_avg_per_method,
        yerr=calinski_harabasz_score_std_per_method, color="blue")
plt.ylabel('Calinski-Harabasz score')
plt.savefig(
    "results/evaluation/all_methods_with_default_parameters/calinski_harabasz_score")

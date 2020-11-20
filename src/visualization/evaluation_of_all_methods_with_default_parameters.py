import joblib
import matplotlib.pyplot as plt
import numpy as np

scores = joblib.load(
    'results/evaluation/all_methods_with_default_parameters.joblib')
methods, iteration_numbers, silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores = zip(
    *scores)

method_abbreviations = list(
    set([method['metadata']['abbreviation'] for method in methods]))
method_abbreviations.sort()

methods_x_axis = [i for i in range(len(method_abbreviations))]


silhouette_scores_by_method = [[] for _ in method_abbreviations]
davies_bouldin_scores_by_method = [[] for _ in method_abbreviations]
calinski_harabasz_scores_by_method = [[] for _ in method_abbreviations]

for i, method in enumerate(methods):
    abbreviation = method_abbreviations.index(
        method['metadata']['abbreviation'])
    silhouette_scores_by_method[abbreviation].append(silhouette_scores[i])
    davies_bouldin_scores_by_method[abbreviation].append(
        davies_bouldin_scores[i])
    calinski_harabasz_scores_by_method[abbreviation].append(
        calinski_harabasz_scores[i])


silhouette_score_avg_per_method = [
    sum(score_list) / len(score_list) for score_list in silhouette_scores_by_method]

davies_bouldin_score_avg_per_method = [
    sum(score_list) / len(score_list) for score_list in davies_bouldin_scores_by_method]

calinski_harabasz_score_avg_per_method = [
    sum(score_list) / len(score_list) for score_list in calinski_harabasz_scores_by_method]

silhouette_score_std_per_method = [
    np.std(score_list) for score_list in silhouette_scores_by_method]
davies_bouldin_score_std_per_method = [
    np.std(score_list) for score_list in davies_bouldin_scores_by_method]
calinski_harabasz_score_std_per_method = [
    np.std(score_list) for score_list in calinski_harabasz_scores_by_method]


plt.rcParams['axes.axisbelow'] = True
plt.grid(axis='y')

plt.bar(methods_x_axis, silhouette_score_avg_per_method,
        yerr=silhouette_score_std_per_method, color='royalblue')
plt.xlabel('Clustering methods')
plt.ylabel('Silhouette score')
plt.xticks(methods_x_axis, method_abbreviations)
plt.savefig('results/evaluation/silhouette_scores.png')

plt.bar(methods_x_axis, davies_bouldin_score_avg_per_method,
        yerr=davies_bouldin_score_std_per_method, color='royalblue')
plt.ylabel('Davies-Bouldin score')
plt.savefig('results/evaluation/davies_bouldin_scores.png')

plt.bar(methods_x_axis, calinski_harabasz_score_avg_per_method,
        yerr=calinski_harabasz_score_std_per_method, color='royalblue')
plt.ylabel('Calinski-Harabasz score')
plt.savefig('results/evaluation/calinski_harabasz_scores.png')

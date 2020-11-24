import numpy as np
import joblib

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# setting distance_threshold=0 ensures we compute the full tree.
model = joblib.load('models/agglomerative_clustering_opt_weighted.joblib')['model']

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel('Number of points in node (or index of point if no parenthesis).')
plt.savefig('results/agglomerative_clustering/optimized_weighted/dendrogram.png')

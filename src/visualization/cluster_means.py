import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/processed/processed.csv')
features = np.array(data.columns)
methods = {
    'agglomerative_clustering': pd.read_csv(
        'results/agglomerative_clustering/optimized/clusters.csv').applymap(str),
    'agglomerative_clustering_opt_weighted': pd.read_csv(
        'results/agglomerative_clustering/optimized_weighted/clusters.csv').applymap(str),
    'mean_shift': pd.read_csv(
        'results/mean_shift/optimized/clusters.csv').applymap(str),
    'mean_shift_weighted': pd.read_csv(
        'results/mean_shift/optimized_weighted/clusters.csv').applymap(str)
}

for method, clustering in methods.items():
    g = sns.PairGrid(
        pd.concat([data, clustering], axis=1),
        x_vars=features,
        y_vars='cluster'
    )
    g.map(sns.barplot)

    plt.savefig(f'results/evaluation/cluster_means/{method}_cluster_means.png')

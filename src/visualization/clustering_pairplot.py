import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/processed/processed.csv')
methods = {
    'agglomerative_clustering': pd.read_csv(
        'results/agglomerative_clustering/optimized/clusters.csv'),
    'mean_shift': pd.read_csv(
        'results/mean_shift/optimized/clusters.csv'),
}

for method, clustering in methods.items():
    sns.pairplot(
        pd.concat([data, clustering], axis=1),
        hue='cluster',
        palette='Set2'
    )
    plt.savefig(f'results/evaluation/{method}_pairplot.png')

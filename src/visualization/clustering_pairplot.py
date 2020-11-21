import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/processed/processed.csv')
clustering = pd.read_csv('results/spectral_clustering/clusters.csv')
data = pd.concat([data, clustering], axis=1)

sns.pairplot(data, hue='cluster', palette='Set2')
plt.savefig('results/evaluation/clustering_pairplot.png')

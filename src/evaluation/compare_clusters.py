import pandas as pd

data = pd.read_csv('data/processed/processed.csv')
ac_clusters = pd.read_csv(
    'results/agglomerative_clustering/optimized/clusters.csv').rename(columns={'cluster': 'ac_cluster'})
birch_clusters = pd.read_csv(
    'results/birch/optimized/clusters.csv').rename(columns={'cluster': 'birch_cluster'})

data = pd.concat([data, ac_clusters, birch_clusters], axis=1)

print(data.groupby(['ac_cluster']).agg(['mean', 'std', 'min', 'max']))
print(data.groupby(['birch_cluster']).agg(['mean', 'std', 'min', 'max']))

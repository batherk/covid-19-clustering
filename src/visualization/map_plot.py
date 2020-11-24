import pandas as pd
import plotly.express as px

data = pd.read_csv('data/clean/clean.csv')
methods = {
    'agglomerative_clustering_opt': pd.read_csv(
        'results/agglomerative_clustering/optimized/clusters.csv'),
    'agglomerative_clustering_opt_weighted': pd.read_csv(
        'results/agglomerative_clustering/optimized_weighted/clusters.csv'),
    'birch_opt': pd.read_csv(
        'results/birch/optimized/clusters.csv'),
    'birch_opt_weighted': pd.read_csv(
        'results/birch/optimized_weighted/clusters.csv')
}
methods['diff_opt'] = pd.DataFrame({
    'cluster': (methods['agglomerative_clustering_opt'] == methods['birch_opt']).all(axis=1)
})
methods['weighted_diff_ac'] = pd.DataFrame({
    'cluster': (methods['agglomerative_clustering_opt'] == methods['agglomerative_clustering_opt_weighted']).all(axis=1)
})
methods['weighted_diff_birch'] = pd.DataFrame({
    'cluster': (methods['birch_opt'] == methods['birch_opt_weighted']).all(axis=1)
})

for method, clustering in methods.items():
    fig = px.choropleth(
        pd.concat([data, clustering], axis=1).applymap(str),
        locations='iso_code',
        color='cluster',
        title=method,
    )
    fig.update_geos(showcountries=True)
    fig.write_image(f'results/evaluation/map_plot/{method}_map_plot.png')

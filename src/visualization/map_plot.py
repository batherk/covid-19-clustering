import pandas as pd
import plotly.express as px

data = pd.read_csv('data/clean/clean.csv')
methods = {
    'agglomerative_clustering': pd.read_csv(
        'results/agglomerative_clustering/optimized/clusters.csv'),
    'agglomerative_clustering_weighted': pd.read_csv(
        'results/agglomerative_clustering/optimized_weighted/clusters.csv'),
    'mean_shift': pd.read_csv(
        'results/mean_shift/optimized/clusters_renamed.csv'),
    'mean_shift_weighted': pd.read_csv(
        'results/mean_shift/optimized_weighted/clusters.csv')
}
methods['diff'] = pd.DataFrame({
    'cluster': (methods['agglomerative_clustering'] == methods['mean_shift']).all(axis=1)
})
methods['weighted_diff_ac'] = pd.DataFrame({
    'cluster': (methods['agglomerative_clustering'] == methods['agglomerative_clustering_weighted']).all(axis=1)
})
methods['weighted_diff_mean_shift'] = pd.DataFrame({
    'cluster': (methods['mean_shift'] == methods['mean_shift_weighted']).all(axis=1)
})

for method, clustering in methods.items():
    fig = px.choropleth(
        pd.concat([data, clustering], axis=1).applymap(str).sort_values(by=['cluster']),
        locations='iso_code',
        color='cluster',
        title=method,
    )
    fig.update_geos(showcountries=True)
    fig.write_image(f'results/evaluation/map_plot/{method}_map_plot.png')

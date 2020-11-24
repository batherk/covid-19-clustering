import pandas as pd
import plotly.express as px

data = pd.read_csv('data/clean/clean.csv')
methods = {
    'agglomerative_clustering': pd.read_csv(
        'results/agglomerative_clustering/optimized/clusters.csv'),
    'birch': pd.read_csv(
        'results/birch/optimized/clusters.csv')
}
methods['diff'] = pd.DataFrame({
    'cluster': (methods['agglomerative_clustering'] == methods['birch']).all(axis=1)
})

for method, clustering in methods.items():
    fig = px.choropleth(
        pd.concat([data, clustering], axis=1).applymap(str),
        locations='iso_code',
        color='cluster',
        title=method,
    )
    fig.update_geos(showcountries=True)
    fig.write_image(f'results/evaluation/{method}_map_plot.png')

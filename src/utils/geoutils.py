import os
import pandas as pd
import plotly.express as px

def plot_clusters_on_world_map(
    basins: list,
    cluster_dict: dict,
    gauge_mapping: dict,
    output_dir: str,
    title_details: str = ""
):
    """
    Plots basin clusters on an interactive world map using Plotly and saves it as an HTML file.

    Args:
        basins (list): List of basin IDs.
        cluster_dict (dict): Mapping of basin_id -> cluster label.
        gauge_mapping (dict): Mapping of basin_id to a dict with 'gauge_lat' and 'gauge_lon'.
        output_dir (str): Directory where the file will be saved. "clusters_world_map.html" will be appended.
        title_details (str, optional): Additional details to append to the title.
    """
    rows = []
    for bid in basins:
        if bid in gauge_mapping:
            rows.append({
                "basin_id": bid,
                "latitude": gauge_mapping[bid]["gauge_lat"],
                "longitude": gauge_mapping[bid]["gauge_lon"],
                "cluster": cluster_dict[bid]
            })
    
    if not rows:
        print("No basins with gauge data found; skipping map plot.")
        return
    
    df = pd.DataFrame(rows)
    df['cluster'] = df['cluster'].astype(int) + 1 # 1-based indexing
    df['cluster'] = df['cluster'].astype(str)
    
    base_title = "Basin Clusters on World Map"
    title = f"{base_title} ({title_details})" if title_details else base_title

    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='cluster',
        hover_name='basin_id',
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set1,
        category_orders={"cluster": sorted(df['cluster'].unique(), key=lambda x: int(x))}
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(217, 217, 217)",
            showcountries=True
        ),
        title=dict(
            x=0.5, 
            xanchor='center'
        ),
        title_font=dict(
            size=20
        )
    )
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clusters_world_map.html")
    fig.write_html(output_path)
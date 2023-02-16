import os
import json

import geopandas as gpd

# Path to the directory where the node and edge files are stored.
ROAD_NETWORK_DIR = "./output/osm_network/"
# Path to the output file.
OUTPUT_FILE = "./output/routing_graph.json"

print("Reading edges")
edges = gpd.read_file(os.path.join(ROAD_NETWORK_DIR, "edges.fgb"))

# Removing duplicate edges.
st_count = edges.groupby(['source_index', 'target_index'])['index'].count()
to_remove = set()
for s, t in st_count.loc[st_count > 1].index:
    dupl = edges.loc[(edges['source_index'] == s) & (edges['target_index'] == t)]
    # Keep only the edge with the smallest travel time.
    tt = dupl['length'] / (dupl['speed'] / 3.6)
    id_min = tt.index[tt.argmin()]
    for i in dupl.index:
        if i != id_min:
            to_remove.add(i)
if to_remove:
    print('Warning. Removing {} duplicate edges.'.format(len(to_remove)))
    edges.drop(labels=to_remove, inplace=True)

print("Creating routing graph")
metro_edges = list()
for _, row in edges.iterrows():
    edge = [
        row["source_index"],
        row["target_index"],
        row['length'] / (row['speed'] / 3.6), # Travel time in seconds.
        ]
    metro_edges.append(edge)

print("Writing data...")
with open(OUTPUT_FILE, "w") as f:
    f.write(json.dumps(metro_edges))

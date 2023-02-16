import os
import json

import geopandas as gpd

# Path to the directory where the node and edge files are stored.
ROAD_NETWORK_DIR = "./output/osm_network/"
# Path to the output file.
OUTPUT_FILE = "./output/routing_graph.json"

print("Reading edges")
edges = gpd.read_file(os.path.join(ROAD_NETWORK_DIR, "edges.fgb"))

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

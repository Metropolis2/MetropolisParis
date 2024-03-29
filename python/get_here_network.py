import os

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import fiona
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge

# Path to shapefile where HERE network data is stored.
STREET_FILE = "./data/here_network/streets/Streets.shp"
# Path to directory where the output files should be stored.
OUTPUT_DIR = "./output/here_network/"
# CRS to use for computation of edge length.
METRIC_CRS = "epsg:2154"
# Max FUNC_CLASS level allowed.
MAX_LEVEL = 4

print("Reading edges...")

streets = fiona.open(STREET_FILE)

# Read records as a DataFrame.
columns = [
    "LINK_ID",
    "ST_NAME",
    "REF_IN_ID",
    "NREF_IN_ID",
    "FUNC_CLASS",
    "FR_SPD_LIM",
    "TO_SPD_LIM",
    "FROM_LANES",
    "TO_LANES",
    "DIR_TRAVEL",
    "geometry",
]
records = list()
for rec in streets:
    # Filter for edges with car access and valid `FUNC_CLASS`.
    if (
        int(rec["properties"]["FUNC_CLASS"]) <= MAX_LEVEL
        and rec["properties"]["AR_AUTO"] == "Y"
        and rec["properties"]["PUB_ACCESS"] == "Y"
    ):
        records.append(rec)
gdf = gpd.GeoDataFrame.from_features(records, crs=streets.crs["init"], columns=columns)

# Add backward roads.
N = len(gdf)
gdf = pd.concat([gdf, gdf.loc[gdf["DIR_TRAVEL"] == "B"]], ignore_index=True)
gdf.loc[N:, "DIR_TRAVEL"] = "T"
gdf.loc[gdf["DIR_TRAVEL"] == "B", "DIR_TRAVEL"] = "F"

# Get source and target.
gdf.loc[gdf["DIR_TRAVEL"] == "F", "source"] = gdf["REF_IN_ID"]
gdf.loc[gdf["DIR_TRAVEL"] == "T", "source"] = gdf["NREF_IN_ID"]
gdf.loc[gdf["DIR_TRAVEL"] == "F", "target"] = gdf["NREF_IN_ID"]
gdf.loc[gdf["DIR_TRAVEL"] == "T", "target"] = gdf["REF_IN_ID"]
gdf["source"] = gdf["source"].astype(int)
gdf["target"] = gdf["target"].astype(int)

# Get number of lanes.
gdf.loc[gdf["DIR_TRAVEL"] == "F", "lanes"] = gdf["FROM_LANES"]
gdf.loc[gdf["DIR_TRAVEL"] == "T", "lanes"] = gdf["TO_LANES"]
invalids = gdf["lanes"] == 0
gdf.loc[invalids, "lanes"] = 1
gdf["lanes"] = gdf["lanes"].astype(int)

# Get speed limit.
gdf.loc[gdf["DIR_TRAVEL"] == "F", "speed"] = gdf["FR_SPD_LIM"].astype(float)
gdf.loc[gdf["DIR_TRAVEL"] == "T", "speed"] = gdf["TO_SPD_LIM"].astype(float)

# Compute road length.
gdf["length"] = gdf.to_crs(METRIC_CRS).length

print("Reversing geometries for backward edges...")

# Reverse geometries for backward roads.
gdf.set_geometry(
    gdf.apply(
        lambda row: row["geometry"]
        if row["DIR_TRAVEL"] != "T"
        else LineString(np.asarray(row["geometry"].coords)[::-1]),
        axis=1,
    ),
    inplace=True,
)

gdf["id"] = np.arange(1, len(gdf) + 1)
gdf.set_index("id", drop=False, inplace=True)
gdf.rename(columns={"ST_NAME": "name", "FUNC_CLASS": "road_type"}, inplace=True)
gdf["road_type"] = gdf["road_type"].astype(int)
gdf.drop(columns=['REF_IN_ID', 'NREF_IN_ID', 'FR_SPD_LIM', 'TO_SPD_LIM', 'FROM_LANES', 'TO_LANES',
                  'DIR_TRAVEL'], inplace=True)

print("Building graph...")

G = nx.DiGraph()
G.add_edges_from(
    map(
        lambda f: (f["properties"]["source"], f["properties"]["target"], f["properties"]),
        gdf.iterfeatures(),
    )
)
# Find the nodes of the largest strongly connected component.
connected_nodes = max(nx.strongly_connected_components(G), key=len)
if len(connected_nodes) < G.number_of_nodes():
    print(
        "Warning: discarding {} nodes disconnected from the main graph".format(
            G.number_of_nodes() - len(connected_nodes)
        )
    )
    G.remove_nodes_from(set(G.nodes).difference(connected_nodes))
    gdf = gdf.loc[gdf["source"].isin(connected_nodes)]
    gdf = gdf.loc[gdf["target"].isin(connected_nodes)]

print("Number of edges: {}".format(len(gdf)))

while False:
    print("Simplifying the network...")
    middle_nodes = filter(
        lambda n: len(G.succ[n]) == 1 and len(G.pred[n]) == 1,
        G.nodes,
    )
    nodes_to_merge = list()
    # Find the nodes that can be merged.
    for node in middle_nodes:
        # The current node has only one in-edge and one out-edge.
        u = next(G.predecessors(node))
        v = next(G.successors(node))
        if u == v:
            # The node is a cul-de-sac, we leave it like that.
            continue
        ux_edge = G[u][node]
        xv_edge = G[node][v]
        if (
            ux_edge["road_type"] == xv_edge["road_type"]
            and ux_edge["lanes"] == xv_edge["lanes"]
            and ux_edge["speed"] == xv_edge["speed"]
        ):
            # The two edges are identical and can be merged.
            nodes_to_merge.append((node, ux_edge["id"], xv_edge["id"]))
    if not nodes_to_merge:
        # No more node can be merged.
        break
    # Merge the nodes.
    nodes_to_remove = set()
    edges_to_remove = set()
    for (node, in_edge_id, out_edge_id) in nodes_to_merge:
        in_edge = gdf.loc[in_edge_id]
        out_edge = gdf.loc[out_edge_id]
        gdf.loc[in_edge_id, "length"] += out_edge["length"]
        gdf.loc[in_edge_id, "target"] = out_edge["target"]
        gdf.loc[in_edge_id, "geometry"] = linemerge(
            MultiLineString([in_edge["geometry"], out_edge["geometry"]])
        )
        assert gdf.loc[in_edge_id, "geometry"].geom_type == "LineString"
        nodes_to_remove.add(node)
        edges_to_remove.add(out_edge_id)
    gdf = gdf.loc[~gdf.index.isin(edges_to_remove)]
    G.remove_nodes_from(nodes_to_remove)
    print("Number of edges: {}".format(len(gdf)))

gdf.drop(columns=["id"], inplace=True)

print('Counting neighbors...')

neighbor_count = pd.Series({n: G.degree[n] for n in G.nodes})
neighbor_count.name = 'neighbor_count'
gdf = gdf.merge(neighbor_count, how='left', left_on='target', right_index=True)

print("Generating the nodes...")

# Get nodes.
all_nodes = set(gdf["source"]).union(set(gdf["target"]))
nodes_gdf = gpd.GeoDataFrame({"id": list(all_nodes)})
points = dict()
for key, row in gdf.iterrows():
    points[row["source"]] = Point(row["geometry"].coords[0])
    points[row["target"]] = Point(row["geometry"].coords[-1])
nodes_gdf.set_geometry([points[node] for node in nodes_gdf["id"]], inplace=True)
nodes_gdf = nodes_gdf.set_crs(epsg=4326)

print('Number of nodes: {}'.format(len(nodes_gdf)))

print("Saving the results...")

nodes_gdf.to_file(os.path.join(OUTPUT_DIR, "here_nodes.fgb"), driver="FlatGeobuf")
gdf.to_file(os.path.join(OUTPUT_DIR, "here_edges.fgb"), driver="FlatGeobuf")

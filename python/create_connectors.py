import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import distance_matrix
from shapely.geometry import LineString

NODE_FILE = "./output/here_network/here_nodes.geojson"
EDGE_FILE = "./output/here_network/here_edges.geojson"
ZONE_FILE = "./data/contours_iris_france/CONTOURS-IRIS.shp"

IDF_DEP = ("75", "77", "78", "91", "92", "93", "94", "95")
METRIC_CRS = "epsg:2154"

# The zone centroids are connected to nodes chosen among the DIST_RANK closest nodes.
DIST_RANK = 25
# Number of lanes on the connectors.
CONNECTOR_LANES = 1
# Speed on the connectors, in km/h.
CONNECTOR_SPEED = 30

# File where the ZONE_ID -> NODE_ID map is saved.
ZONE_ID_FILE = "./output/here_network/zone_id_map.csv"
OUTPUT_NODE_FILE = "./output/here_network/here_nodes_with_zones.geojson"
OUTPUT_EDGE_FILE = "./output/here_network/here_edges_with_connectors.geojson"

print("Reading network files")

nodes = gpd.read_file(NODE_FILE)
edges = gpd.read_file(EDGE_FILE)

edges.to_crs(METRIC_CRS, inplace=True)

nodes.to_crs(METRIC_CRS, inplace=True)
nodes["x"] = nodes.centroid.x
nodes["y"] = nodes.centroid.y
node_coords = nodes[["x", "y"]].to_numpy()

print("Computing node in/out-degree")

# The in-degree of a node is the number of edges whose target node is this node.
# The out-degree of a node is the number of edges whose source node is this node.
# We compute a in/out-degree weighted by number of lanes.
in_degrees = edges.groupby("target")["lanes"].sum()
in_degrees.name = "in_degree"
out_degrees = edges.groupby("source")["lanes"].sum()
out_degrees.name = "out_degree"
nodes = nodes.merge(in_degrees, left_on="id", right_index=True, how="left")
nodes = nodes.merge(out_degrees, left_on="id", right_index=True, how="left")
nodes["in_degree"] = nodes["in_degree"].fillna(0).astype(np.int64)
nodes["out_degree"] = nodes["out_degree"].fillna(0).astype(np.int64)

# Candidate nodes for in/out connectors.
in_nodes = nodes.loc[nodes["in_degree"] > 0]
out_nodes = nodes.loc[nodes["out_degree"] > 0]

print("Reading zones")

zones = gpd.read_file(ZONE_FILE)
zones = zones.loc[zones["INSEE_COM"].str[:2].isin(IDF_DEP)].copy()
zones["CODE_IRIS"] = zones["CODE_IRIS"].astype(np.int64)

# We ensure that zone ids are all distinct from node ids.
max_node_id = nodes["id"].max()
zones.set_index(np.arange(max_node_id + 1, max_node_id + len(zones) + 1), inplace=True)
zones.index.name = "node_id"
zones["CODE_IRIS"].to_csv(ZONE_ID_FILE)

zones.to_crs(METRIC_CRS, inplace=True)
zones["x"] = zones.centroid.x
zones["y"] = zones.centroid.y

print("Building connectors")

quantile = DIST_RANK / len(nodes)

gradiants = [
    {"name": "N", "filter": lambda theta: (theta > -45.0) & (theta < 45.0)},
    {"name": "W", "filter": lambda theta: (theta > -135.0) & (theta < -45.0)},
    {"name": "S", "filter": lambda theta: (theta < -135.0) | (theta > 135.0)},
    {"name": "E", "filter": lambda theta: (theta > 45.0) & (theta < 135.0)},
]

out_connectors = list()
in_connectors = list()
next_id = edges["id"].max() + 1
road_type = edges["road_type"].max() + 1

n = len(zones) // 100
for i, (zone_id, zone) in enumerate(zones.iterrows()):
    if i % n == 0:
        print("{} %".format(i // n))
    distances = distance_matrix(node_coords, np.array([[zone["x"], zone["y"]]]))
    dist_threshold = np.quantile(distances, quantile)

    mask = distances <= dist_threshold
    candidate_nodes = nodes.loc[mask].copy()
    candidate_nodes["distance"] = distances[mask]
    candidate_nodes.sort_values("distance", inplace=True)

    dx = candidate_nodes["x"] - zone["x"]
    dy = candidate_nodes["y"] - zone["y"]
    candidate_nodes["angle"] = np.degrees(np.arctan2(dy, dx))

    for gradiant in gradiants:
        mask = gradiant["filter"](candidate_nodes["angle"])
        if np.all(~mask):
            # No candidate node in this gradiant.
            continue
        # Select the closest node among all the nodes that maximize in-degree.
        in_node_id = candidate_nodes.loc[mask, "in_degree"].idxmax()
        in_node = candidate_nodes.loc[in_node_id]
        if in_node["in_degree"] > 0:
            geom = LineString([zone["geometry"].centroid.coords[0], in_node["geometry"].coords[0]])
            connector = {
                "id": next_id,
                "name": "Out connector {} -> {}".format(zone["CODE_IRIS"], in_node_id),
                "road_type": road_type,
                "lanes": CONNECTOR_LANES,
                "speed": CONNECTOR_SPEED,
                "source": zone_id,
                "target": in_node_id,
                "LINK_ID": 0,
                "geometry": geom,
                "length": geom.length,
            }
            next_id += 1
            out_connectors.append(connector)

        # Select the closest node among all the nodes that maximize out-degree.
        out_node_id = candidate_nodes.loc[mask, "out_degree"].idxmax()
        out_node = candidate_nodes.loc[out_node_id]
        if out_node["out_degree"] > 0:
            geom = LineString([out_node["geometry"].coords[0], zone["geometry"].centroid.coords[0]])
            connector = {
                "id": next_id,
                "name": "Out connector {} -> {}".format(zone["CODE_IRIS"], out_node_id),
                "road_type": road_type,
                "lanes": CONNECTOR_LANES,
                "speed": CONNECTOR_SPEED,
                "source": out_node_id,
                "target": zone_id,
                "LINK_ID": 0,
                "geometry": geom,
                "length": geom.length,
            }
            next_id += 1
            in_connectors.append(connector)

print("Saving edges")

out_connectors = gpd.GeoDataFrame(out_connectors, crs=edges.crs)
in_connectors = gpd.GeoDataFrame(in_connectors, crs=edges.crs)
edges = gpd.GeoDataFrame(pd.concat((edges, out_connectors, in_connectors), ignore_index=True))
edges.to_file(OUTPUT_EDGE_FILE, driver='GeoJSON')

print("Saving nodes")

nodes.drop(columns=['x', 'y', 'in_degree', 'out_degree'], inplace=True)
nodes['is_zone'] = False

zones = zones['geometry'].reset_index().rename(columns={'node_id': 'id'})
zones['geometry'] = zones.centroid
zones['is_zone'] = True

nodes = gpd.GeoDataFrame(pd.concat((nodes, zones), ignore_index=True))
nodes.to_file(OUTPUT_NODE_FILE, driver='GeoJSON')

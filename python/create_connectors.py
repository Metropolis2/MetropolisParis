import os

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import distance_matrix
from shapely.geometry import LineString

# Path to the file with the pre-processed nodes.
NODE_FILE = "./output/osm_network/osm_nodes.fgb"
# Path to the file with the pre-processed edges.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# Path to the IRIS shapefile with the geometries of the origin / destination zones.
ZONE_FILE = "./data/contours_iris_france/CONTOURS-IRIS.shp"
# Départements of the study area, used to filter the IRIS zones.
DEPARTEMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]
# CRS to use for the output file. Should be a projected coordinate system.
METRIC_CRS = "epsg:2154"
# The zone centroids are connected to nodes chosen among the DIST_RANK closest nodes.
DIST_RANK = 20
# Minimum number of nodes that needs to be in a zone to use the medoid as representative point.
MIN_NB_NODES = 5
# Number of lanes on the connectors.
CONNECTOR_LANES = 1
# Speed on the connectors, in km/h.
CONNECTOR_SPEED = 30
# File where the ZONE_ID -> NODE_ID map is saved.
ZONE_ID_FILE = "./output/zone_id_map_osm.csv"
# Path to the directory where the output node and edge files are stored.
OUTPUT_DIR = "./output/osm_network/"

print("Reading network files")

nodes = gpd.read_file(NODE_FILE)
edges = gpd.read_file(EDGE_FILE)

edges.to_crs(METRIC_CRS, inplace=True)

nodes.to_crs(METRIC_CRS, inplace=True)
nodes["x"] = nodes.centroid.x
nodes["y"] = nodes.centroid.y
node_coords = nodes[["x", "y"]].to_numpy()

# Node ande edge ids should start at 0.
nodes.reset_index(drop=True, inplace=True)
edges.reset_index(drop=True, inplace=True)
edges.drop(columns="id", inplace=True)
node_id_map = nodes["id"].to_frame().reset_index().set_index("id")
edges = edges.merge(node_id_map, left_on="source", right_index=True).rename(
    columns={"index": "source_index"}
)
edges = edges.merge(node_id_map, left_on="target", right_index=True).rename(
    columns={"index": "target_index"}
)

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
zones = zones.loc[zones["INSEE_COM"].str[:2].isin(DEPARTEMENTS)].copy()
zones.drop(columns=["INSEE_COM", "NOM_COM", "IRIS", "NOM_IRIS", "TYP_IRIS"], inplace=True)
zones["CODE_IRIS"] = zones["CODE_IRIS"].astype(np.int64)

# Zone ids start at the largest node id + 1.
next_node_id = nodes.index.max() + 1
in_zone_ids = np.arange(next_node_id, next_node_id + len(zones))
zones.set_index(in_zone_ids, inplace=True)

zone_id_map = zones["CODE_IRIS"].copy()
zone_id_map = pd.concat((zone_id_map, zone_id_map), ignore_index=True)
zone_id_map.index = np.arange(next_node_id, next_node_id + 2 * len(zones))
zone_id_map.index.name = "node_id"
zone_id_map = zone_id_map.to_frame()
zone_id_map["in"] = True
zone_id_map.loc[next_node_id + len(zones) :, "in"] = False
zone_id_map.to_csv(ZONE_ID_FILE, index=True)

zones.to_crs(METRIC_CRS, inplace=True)
zones["x"] = zones.centroid.x
zones["y"] = zones.centroid.y
zone_coords = zones[["x", "y"]].to_numpy()

# Simplify the polygons to speed-up computation of distance.
zones["geometry"] = zones["geometry"].simplify(50.0)

nb_zones = len(zones)

print("Building connectors")

quantile = DIST_RANK / len(nodes)

gradiants = [
    {"name": "N", "filter": lambda theta: (theta > -45.0) & (theta < 45.0)},
    {"name": "W", "filter": lambda theta: (theta > -135.0) & (theta < -45.0)},
    {"name": "S", "filter": lambda theta: (theta < -135.0) | (theta > 135.0)},
    {"name": "E", "filter": lambda theta: (theta > 45.0) & (theta < 135.0)},
]

virtual_nodes = list()
representative_points = list()
out_connectors = list()
in_connectors = list()
# Road type id for connectors is 0 so that road types are indexed starting at 0.
road_type = 0

n = nb_zones // 99
for i, (zone_id, zone) in enumerate(zones.iterrows()):
    if i % n == 0:
        print("{} %".format(i // n))
    # Compute distance of all nodes to the zone Polygon.
    distances = nodes["geometry"].distance(zone["geometry"])

    # Find the representative point of the zone.
    zone_mask = distances == 0
    if np.sum(zone_mask) > MIN_NB_NODES:
        # There are some nodes in the zone: the representative point is the medoid of this points.
        nodes_in_zone = nodes.loc[zone_mask]
        dist_matrix = distance_matrix(
            nodes_in_zone[["x", "y"]].values, nodes_in_zone[["x", "y"]].values
        )
        index = dist_matrix.sum(axis=0).argmin()
        rep_point = nodes_in_zone.iloc[index]["geometry"]
        representative_points.append([rep_point, True])
        mean_distances = dist_matrix.mean(axis=0)
        zone_mean_distance = mean_distances.mean()
    else:
        # Use the centroid of the Polygon as representative point.
        rep_point = zone["geometry"].centroid
        representative_points.append([rep_point, False])
        mean_distances = np.array([])
        # When there is no point in the zone, we assume that the average distance between points in
        # the zone is the square root of its area divided by pi.
        # If the zone is a circle, this means that, on average, an agent located in the zone needs
        # to travel a distance equal to the radius of the circle to exit the zone.
        zone_mean_distance = np.sqrt(zone["geometry"].area / np.pi)
    virtual_nodes.append(
        {
            "id": zone_id,
            "geometry": rep_point,
        }
    )

    # Find the nodes that are as close as the DIST_RANK closest node.
    dist_threshold = np.quantile(distances, quantile)
    mask = distances <= dist_threshold
    candidate_nodes = nodes.loc[mask].copy()

    # Compute the average distance to the other points in the zone.
    candidate_nodes["mean_distance"] = np.inf
    if mean_distances.size > 0:
        candidate_nodes.loc[zone_mask, "mean_distance"] = mean_distances

    # The connected nodes are in priority those inside the zone Polygon. If multiple candidate nodes
    # are inside the zone Polygon, then the furthest away from the representative point are selected
    # in priority.
    candidate_nodes["zone_distance"] = distances[mask]
    candidate_nodes["rep_point_distance"] = candidate_nodes["geometry"].distance(rep_point)
    candidate_nodes.sort_values(
        ["zone_distance", "rep_point_distance"], ascending=[True, False], inplace=True
    )

    # Compute the angle between the representative point and the candidate nodes.
    dx = candidate_nodes["x"] - rep_point.x
    dy = candidate_nodes["y"] - rep_point.y
    candidate_nodes["angle"] = np.degrees(np.arctan2(dy, dx))

    for gradiant in gradiants:
        mask = gradiant["filter"](candidate_nodes["angle"])
        if not np.any(mask):
            # No candidate node in this gradiant.
            continue
        # Take the first node in the filtered DataFrame, i.e., the nodes that maximizes
        # `rep_point_distance` among the nodes that minimizes `zone_distance`.
        valid_nodes = candidate_nodes.loc[mask & (candidate_nodes["in_degree"] > 0)]
        if len(valid_nodes):
            node = valid_nodes.iloc[0]
            node_id = node.name
            geom = LineString([(node["x"], node["y"]), (rep_point.x, rep_point.y)])
            if node["zone_distance"] > 0.0 or mean_distances.size == 0:
                # The connected node is not in the zone. Its distance is the distance to the zone
                # plus the average distance between points in the zone.
                length = node["zone_distance"] + zone_mean_distance
            else:
                # The connected node is in the zone. Its distance is the average distance to other
                # points in the zone.
                length = node["mean_distance"]
            connector = {
                "name": "In connector {} -> {}".format(node["id"], zone["CODE_IRIS"]),
                "road_type": road_type,
                "lanes": CONNECTOR_LANES,
                "speed": CONNECTOR_SPEED,
                "source": node["id"],
                "target": zone["CODE_IRIS"],
                "source_index": node_id,
                "target_index": zone_id,
                "geometry": geom,
                "length": length,
            }
            in_connectors.append(connector)

        valid_nodes = candidate_nodes.loc[mask & (candidate_nodes["out_degree"] > 0)]
        if len(valid_nodes):
            node = valid_nodes.iloc[0]
            node_id = node.name
            geom = LineString([(rep_point.x, rep_point.y), (node["x"], node["y"])])
            if node["zone_distance"] > 0.0 or mean_distances.size == 0:
                length = node["zone_distance"] + zone_mean_distance
            else:
                length = node["mean_distance"]
            connector = {
                "name": "Out connector {} -> {}".format(zone["CODE_IRIS"], node["id"]),
                "road_type": road_type,
                "lanes": CONNECTOR_LANES,
                "speed": CONNECTOR_SPEED,
                "source": zone["CODE_IRIS"],
                "target": node["id"],
                # For out connectors, we use the id of the outgoing node of the zone.
                "source_index": zone_id + nb_zones,
                "target_index": node_id,
                "geometry": geom,
                "length": length,
                "neighbor_count": 0,
            }
            out_connectors.append(connector)


print(
    "Created {} in-connectors and {} out-connectors".format(len(in_connectors), len(out_connectors))
)

nb_centroids = len(zones) - sum(rep[1] for rep in representative_points)
print("Number of zones with less than {} nodes: {}".format(MIN_NB_NODES, nb_centroids))

print("Saving edges")

out_connectors = gpd.GeoDataFrame(out_connectors, crs=edges.crs)
print(
    "Distribution of number of out connectors:\n{}".format(
        out_connectors["source_index"].value_counts().value_counts()
    )
)
in_connectors = gpd.GeoDataFrame(in_connectors, crs=edges.crs)
print(
    "Distribution of number of in connectors:\n{}".format(
        in_connectors["target_index"].value_counts().value_counts()
    )
)
connectors = pd.concat((out_connectors, in_connectors), ignore_index=True)
edges = gpd.GeoDataFrame(pd.concat((edges, connectors), ignore_index=True)).reset_index()
edges.to_file(os.path.join(OUTPUT_DIR, "edges.fgb"), driver="FlatGeobuf")

print("Saving nodes")

nodes.drop(columns=["x", "y", "in_degree", "out_degree"], inplace=True)
nodes["is_zone"] = False

zones = zones["geometry"].to_frame().reset_index(drop=True)
representative_points = [rep[0] for rep in representative_points]
zones["geometry"] = representative_points
zones["is_zone"] = True
# Double the zones for in and out nodes.
zones = gpd.GeoDataFrame(pd.concat((zones, zones), ignore_index=True))
zones["id"] = 0

nodes = gpd.GeoDataFrame(pd.concat((nodes, zones), ignore_index=True)).reset_index()
nodes.to_file(os.path.join(OUTPUT_DIR, "nodes.fgb"), driver="FlatGeobuf")

import sys
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import osmium
import networkx as nx
from geojson import Point, LineString, Feature, FeatureCollection
from haversine import haversine_vector, Unit
from shapely.ops import linemerge
from shapely.geometry import MultiLineString
from collections import defaultdict

# Path to the OSM PBF file.
OSM_FILE = "./data/osm/paris-filtered.osm.pbf"
# Path to the FlatGeobuf file where nodes should be stored.
NODE_FILE = "./output/osm_network/osm_nodes.fgb"
# Path to the FlatGeobuf file where edges should be stored.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# List of highway tags to consider.
# See https://wiki.openstreetmap.org/wiki/Key:highway
VALID_HIGHWAYS = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    'residential',
    'living_street',
    'unclassified',
    'road',
    'service',
)
# Road type id to use for each highway tag.
ROADTYPE_TO_ID = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 7,
    "motorway_link": 8,
    "trunk_link": 9,
    "primary_link": 10,
    "secondary_link": 11,
    "tertiary_link": 12,
    "living_street": 13,
    "road": 14,
    "service": 15,
}
# Default number of lanes when unspecified.
DEFAULT_LANES = {
    "motorway": 2,
    "trunk": 2,
    "primary": 1,
    "secondary": 1,
    "tertiary": 1,
    "unclassified": 1,
    "residential": 1,
    "motorway_link": 1,
    "trunk_link": 1,
    "primary_link": 1,
    "secondary_link": 1,
    "tertiary_link": 1,
    "living_street": 1,
    "road": 1,
    "service": 1,
}
# Default speed, in km/h, when unspecified.
DEFAULT_SPEED = {
    "motorway": 130,
    "trunk": 110,
    "primary": 80,
    "secondary": 80,
    "tertiary": 80,
    "unclassified": 30,
    "residential": 30,
    "motorway_link": 90,
    "trunk_link": 70,
    "primary_link": 50,
    "secondary_link": 50,
    "tertiary_link": 50,
    "living_street": 20,
    "road": 50,
    "service": 30,
}


def valid_way(way):
    return len(way.nodes) > 1 and way.tags.get("highway") in VALID_HIGHWAYS


class NodeReader(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.all_nodes = set()
        self.nodes = dict()
        self.counter = 0

    def way(self, way):
        if not valid_way(way):
            return
        self.handle_way(way)

    def handle_way(self, way):
        # Always add source and origin node.
        self.add_node(way.nodes[0])
        self.add_node(way.nodes[-1])
        self.all_nodes.add(way.nodes[0])
        self.all_nodes.add(way.nodes[-1])
        # Add the other nodes if they were already explored, i.e., they
        # intersect with another road.
        for i in range(1, len(way.nodes) - 1):
            node = way.nodes[i]
            if node in self.all_nodes:
                self.add_node(node)
            self.all_nodes.add(node)

    def add_node(self, node):
        if node.ref in self.nodes:
            # Node was already added.
            return
        if node.location.valid():
            self.nodes[node.ref] = Feature(
                geometry=Point((node.lon, node.lat)),
                properties={"id": self.counter, "osm_id": node.ref},
            )
            self.counter += 1


class Writer(osmium.SimpleHandler):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.edges = list()
        self.counter = 0

    def way(self, way):
        self.add_way(way)

    def add_way(self, way):

        if not valid_way(way):
            return

        road_type = way.tags.get("highway", None)
        road_type_id = ROADTYPE_TO_ID[road_type]

        name = (
            way.tags.get("name", "") or way.tags.get("addr:street", "") or way.tags.get("ref", "")
        )
        if len(name) > 50:
            name = name[:47] + "..."

        oneway = (
            way.tags.get("oneway", "no") == "yes" or way.tags.get("junction", "") == "roundabout"
        )

        # Find maximum speed if available.
        maxspeed = way.tags.get("maxspeed", "")
        speed = None
        back_speed = None
        if maxspeed == "FR:walk":
            speed = 20
        elif maxspeed == "FR:urban":
            speed = 50
        elif maxspeed == "FR:rural":
            speed = 80
        else:
            try:
                speed = float(maxspeed)
            except ValueError:
                pass
        if not oneway:
            try:
                speed = float(way.tags.get("maxspeed:forward", "0")) or speed
            except ValueError:
                pass
            try:
                back_speed = float(way.tags.get("maxspeed:backward", "0")) or speed
            except ValueError:
                pass
        if speed is None:
            speed = DEFAULT_SPEED.get(road_type, 50)
        if back_speed is None:
            back_speed = DEFAULT_SPEED.get(road_type, 50)

        # Find number of lanes if available.
        lanes = None
        back_lanes = None
        if oneway:
            try:
                lanes = int(way.tags.get("lanes", ""))
            except ValueError:
                pass
            else:
                lanes = max(lanes, 1)
        else:
            try:
                lanes = (
                    int(way.tags.get("lanes:forward", "0")) or int(way.tags.get("lanes", "")) // 2
                )
            except ValueError:
                pass
            else:
                lanes = max(lanes, 1)
            try:
                back_lanes = (
                    int(way.tags.get("lanes:backward", "0")) or int(way.tags.get("lanes", "")) // 2
                )
            except ValueError:
                pass
            else:
                back_lanes = max(back_lanes, 1)
        if lanes is None:
            lanes = DEFAULT_LANES.get(road_type, 1)
        if back_lanes is None:
            back_lanes = DEFAULT_LANES.get(road_type, 1)

        for i, node in enumerate(way.nodes):
            if node.ref in self.nodes:
                source = i
                break
        else:
            # No node of the way is in the nodes.
            return

        j = source + 1
        for i, node in enumerate(list(way.nodes)[j:]):
            if node.ref in self.nodes:
                target = j + i
                self.add_edge(
                    way,
                    source,
                    target,
                    oneway,
                    name,
                    road_type_id,
                    lanes,
                    back_lanes,
                    speed,
                    back_speed,
                )
                source = target

    def add_edge(
        self, way, source, target, oneway, name, road_type, lanes, back_lanes, speed, back_speed
    ):
        source_id = self.nodes[way.nodes[source].ref].properties["id"]
        target_id = self.nodes[way.nodes[target].ref].properties["id"]
        if source_id == target_id:
            # Self-loop.
            return

        # Create a geometry of the road.
        coords = list()
        for i in range(source, target + 1):
            if way.nodes[i].location.valid():
                coords.append((way.nodes[i].lon, way.nodes[i].lat))
        geometry = LineString(coords)
        if not oneway:
            back_geometry = LineString(coords[::-1])

        edge_id = self.counter
        self.counter += 1
        if not oneway:
            back_edge_id = self.counter
            self.counter += 1

        # Compute length in kilometers.
        length = haversine_vector(coords[:-1], coords[1:], Unit.KILOMETERS).sum()

        self.edges.append(
            Feature(
                geometry=geometry,
                properties={
                    "id": edge_id,
                    "name": name,
                    "road_type": road_type,
                    "lanes": lanes,
                    "length": length,
                    "speed": speed,
                    "source": source_id,
                    "target": target_id,
                    "osm_id": way.id,
                },
            )
        )

        if not oneway:
            self.edges.append(
                Feature(
                    geometry=back_geometry,
                    properties={
                        "id": back_edge_id,
                        "name": name,
                        "road_type": road_type,
                        "lanes": back_lanes,
                        "length": length,
                        "speed": back_speed,
                        "source": target_id,
                        "target": source_id,
                        "osm_id": way.id,
                    },
                )
            )

    def post_process(self, simplify=False):
        node_collection = FeatureCollection(
            list(self.nodes.values()),
            crs={"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        )
        edge_collection = FeatureCollection(
            self.edges,
            crs={"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        )
        nodes = gpd.GeoDataFrame.from_features(node_collection)
        edges = gpd.GeoDataFrame.from_features(edge_collection)
        edges = self.count_neighbors(edges)

        print("Building graph...")

        G = nx.DiGraph()
        G.add_edges_from(
            map(
                lambda f: (f["properties"]["source"], f["properties"]["target"], f["properties"]),
                edges.iterfeatures(),
            )
        )
        # Find the nodes of the largest weakly connected component.
        connected_nodes = max(nx.weakly_connected_components(G), key=len)
        if len(connected_nodes) < G.number_of_nodes():
            print(
                "Warning: discarding {} nodes disconnected from the main graph".format(
                    G.number_of_nodes() - len(connected_nodes)
                )
            )
            G.remove_nodes_from(set(G.nodes).difference(connected_nodes))
            edges = edges.loc[edges["source"].isin(connected_nodes)]
            edges = edges.loc[edges["target"].isin(connected_nodes)].copy()
            nodes = nodes.loc[nodes["id"].isin(connected_nodes)].copy()

        print("Number of edges: {}".format(len(edges)))


        if simplify:
            nodes, edges = self.simplify(nodes, edges)
        self.nodes = nodes
        self.edges = edges

    def count_neighbors(self, edges):
        in_neighbors = edges.groupby(["target"])["source"].unique()
        out_neighbors = edges.groupby(["source"])["target"].unique()
        node_neighbors = pd.DataFrame({"in": in_neighbors, "out": out_neighbors})

        def merge_lists(row):
            if row["in"] is np.nan:
                in_set = set()
            else:
                in_set = set(row["in"])
            if row["out"] is np.nan:
                out_set = set()
            else:
                out_set = set(row["out"])
            return in_set.union(out_set)

        node_neighbors = node_neighbors.apply(merge_lists, axis=1)
        neighbor_counts = node_neighbors.apply(lambda x: len(x))
        neighbor_counts.name = "neighbor_count"
        edges = edges.merge(neighbor_counts, how="left", left_on="target", right_index=True)
        assert not edges["neighbor_count"].isna().any()
        return edges

    def simplify(self, nodes, edges):
        variables = ["road_type", "lanes", "speed"]
        indices_map = dict()
        reverse_indices_map = defaultdict(list)
        in_neighbors = edges.groupby(["target"])["source"].unique()
        out_neighbors = edges.groupby(["source"])["target"].unique()
        node_neighbors = pd.DataFrame({"in": in_neighbors, "out": out_neighbors})
        node_neighbors.dropna(inplace=True)
        node_neighbors = node_neighbors.apply(lambda x: set(x["in"]).union(set(x["out"])), axis=1)
        neighbor_counts = node_neighbors.apply(lambda x: len(x))
        node_neighbors = node_neighbors.loc[neighbor_counts == 2]
        all_in_edges = edges.loc[edges["target"].isin(node_neighbors.index)].copy()
        all_in_indices = all_in_edges.groupby("target").indices
        all_out_edges = edges.loc[edges["source"].isin(node_neighbors.index)].copy()
        all_out_indices = all_out_edges.groupby("source").indices
        for node, neighbors in node_neighbors.items():
            # The current node has no route choice (it is a node in the middle
            # of a one-way or two-way road): we can remove it if the incoming
            # and outgoing edges are similar.
            in_indices = all_in_edges.iloc[all_in_indices[node]].index
            out_indices = all_out_edges.iloc[all_out_indices[node]].index
            in_indices = in_indices.map(lambda idx: indices_map.get(idx, idx))
            out_indices = out_indices.map(lambda idx: indices_map.get(idx, idx))
            in_edges = edges.loc[in_indices]
            out_edges = edges.loc[out_indices]
            neighbors = list(neighbors)
            # Case 1.
            in_edge = in_edges.loc[in_edges["source"] == neighbors[0]]
            out_edge = out_edges.loc[out_edges["target"] == neighbors[1]]
            if len(in_edge) == 1 and len(out_edge) == 1:
                in_edge = in_edge.iloc[0]
                out_edge = out_edge.iloc[0]
                if (
                    (in_edge[variables] == out_edge[variables])
                    | (in_edge[variables].isnull() & out_edge[variables].isnull())
                ).all():
                    # Merge the two edges.
                    self.merge_edges(in_edge, out_edge, edges)
                    indices_map[out_edge.name] = in_edge.name
                    reverse_indices_map[in_edge.name].append(out_edge.name)
                    for old_node in reverse_indices_map[out_edge.name]:
                        indices_map[old_node] = in_edge.name
            # Case 2.
            in_edge = in_edges.loc[in_edges["source"] == neighbors[1]]
            out_edge = out_edges.loc[out_edges["target"] == neighbors[0]]
            if len(in_edge) == 1 and len(out_edge) == 1:
                in_edge = in_edge.iloc[0]
                out_edge = out_edge.iloc[0]
                if (
                    (in_edge[variables] == out_edge[variables])
                    | (in_edge[variables].isnull() & out_edge[variables].isnull())
                ).all():
                    # Merge the two edges.
                    self.merge_edges(in_edge, out_edge, edges)
                    indices_map[out_edge.name] = in_edge.name
                    reverse_indices_map[in_edge.name].append(out_edge.name)
                    for old_node in reverse_indices_map[out_edge.name]:
                        indices_map[old_node] = in_edge.name
        node_ids = set(edges["source"].values).union(edges["target"].values)
        nodes = nodes.loc[nodes["id"].isin(node_ids)]
        return (nodes, edges)

    def merge_edges(self, in_edge, out_edge, edges):
        edges.loc[in_edge.name, "length"] += out_edge["length"]
        edges.loc[in_edge.name, "target"] = out_edge["target"]
        edges.loc[in_edge.name, "osm_id"] = 0
        edges.loc[in_edge.name, "neighbor_count"] = out_edge["neighbor_count"]
        edges.loc[in_edge.name, "geometry"] = linemerge(
            MultiLineString([in_edge["geometry"], out_edge["geometry"]])
        )
        assert edges.loc[in_edge.name, "geometry"].geom_type == "LineString"
        edges.drop(index=out_edge.name, inplace=True)
        return edges

    def write_edges(self, filename):
        self.edges.to_file(filename, driver="FlatGeobuf", crs="epsg:4326")

    def write_nodes(self, filename):
        self.nodes.to_file(filename, driver="FlatGeobuf", crs="epsg:4326")


if __name__ == "__main__":

    # File does not exists or is not in the same folder as the script.
    if not os.path.exists(OSM_FILE):
        print("File not found: {}".format(OSM_FILE))
        sys.exit(0)

    h = NodeReader()

    print("Finding nodes...")
    h.apply_file(OSM_FILE, locations=True, idx="flex_mem")

    g = Writer(h.nodes)

    print("Reading OSM data...")
    g.apply_file(OSM_FILE, locations=True, idx="flex_mem")

    print("Post-processing...")
    g.post_process(simplify=False)

    print("Found {} nodes and {} edges.".format(len(g.nodes), len(g.edges)))

    print("Writing edges...")
    g.write_edges(EDGE_FILE)

    print("Writing nodes...")
    g.write_nodes(NODE_FILE)

    print("Done!")

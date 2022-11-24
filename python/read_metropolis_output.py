import json

import geopandas as gpd
import folium
from folium.vector_layers import Circle, PolyLine
import matplotlib as mpl
from matplotlib.colors import to_hex
import zstandard as zstd

# Path to the output agent-results file is stored
AGENT_RESULTS_FILENAME = "./output/server_runs/agent_results3.json.zst"
# Path to the file where edge geometries are stored.
EDGE_FILENAME = "./output/here_network/edges.fgb"


def get_agent_results():
    dctx = zstd.ZstdDecompressor()
    with open(AGENT_RESULTS_FILENAME, 'br') as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data


def get_edges():
    edges = gpd.read_file(EDGE_FILENAME)
    # TODO: Remove this when the input of Metropolis is fixed.
    edges["index"] = edges.index
    return edges


def get_agent_map(agent, edges):
    assert agent["mode_results"]["type"] == "Road"
    route = agent["mode_results"]["value"]["route"]
    edges_taken = set(e["edge"] for e in route)
    edges_taken = edges.loc[edges["index"].isin(edges_taken)].set_index("index")
    centroids = edges_taken.centroid.to_crs("epsg:4326")
    edges_taken.to_crs("epsg:4326", inplace=True)
    edges_taken["fftt"] = edges_taken["length"] / (edges_taken["speed"] / 3.6)

    colormap = mpl.colormaps["RdYlGn"]

    mean_location = [centroids.y.mean(), centroids.x.mean()]
    origin_coords = edges_taken.loc[route[0]["edge"], "geometry"].coords[0][::-1]
    destination_coords = edges_taken.loc[route[-1]["edge"], "geometry"].coords[-1][::-1]

    m = folium.Map(location=mean_location, zoom_start=13, tiles="OpenStreetMap")
    Circle(
        location=origin_coords,
        radius=30,
        tooltip="Origin",
        opacity=0.7,
        fill_opacity=0.7,
        color="#E52424",
        fill_color="#E52424",
    ).add_to(m)
    Circle(
        location=destination_coords,
        radius=30,
        tooltip="Destination",
        opacity=0.7,
        fill_opacity=0.7,
        color="#245CE5",
        fill_color="#E52424",
    ).add_to(m)
    for leg in route:
        edge = edges_taken.loc[leg["edge"]]
        edge_coords = list(edge["geometry"].coords)
        edge_coords = [p[::-1] for p in edge_coords]
        edge_tt = leg["edge_exit"] - leg["edge_entry"]
        congestion = edge["fftt"] / edge_tt
        color = to_hex(colormap(congestion))
        tooltip = "From {} to {}<br>Travel time: {}<br>Free-flow tt: {}".format(
            get_time_str(leg["edge_entry"]),
            get_time_str(leg["edge_exit"]),
            get_tt_str(edge_tt),
            get_tt_str(edge["fftt"]),
        )
        PolyLine(
            locations=edge_coords, tooltip=tooltip, opacity=0.5, color=color, weight=10
        ).add_to(m)
    return m


def get_time_str(seconds_after_midnight):
    t = round(seconds_after_midnight)
    hours = t // 3600
    remainder = t % 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)


def get_tt_str(seconds):
    t = round(seconds)
    minutes = int(t // 60)
    seconds = t % 60
    return "{:01}'{:02}''".format(minutes, seconds)

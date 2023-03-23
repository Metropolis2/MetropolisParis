from collections import defaultdict
import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.vector_layers import Circle, PolyLine
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import zstandard as zstd

# Path to the input agent file.
AGENT_INPUT_FILENAME = "./output/server_runs/23/agents.json"
# Path to the output agent-results file.
AGENT_RESULTS_FILENAME = "./output/server_runs/23/agent_results.json.zst"
# Path to the output weights results file.
WEIGHT_RESULTS_FILENAME = "./output/server_runs/23/weight_results.json.zst"
# Path to the output skim results file.
SKIM_RESULTS_FILENAME = "./output/server_runs/23/skim_results.json.zst"
# Path to the directory where OD pairs travel times are stored.
TT_DIR = "./output/ahmed/"
# Path to the directory where EGT data is stored.
EGT_DIR = "~/GitRepositories/mode_choice_reg/data/egt/Format_csv/"
# Path to the file where edge geometries are stored.
EDGE_FILENAME = "./output/here_network/edges.fgb"


def get_agents():
    with open(AGENT_INPUT_FILENAME, "r") as f:
        agents = json.load(f)
    return agents


def get_agent_results():
    dctx = zstd.ZstdDecompressor()
    with open(AGENT_RESULTS_FILENAME, "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data


def get_weight_results():
    dctx = zstd.ZstdDecompressor()
    with open(WEIGHT_RESULTS_FILENAME, "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data


def get_skim_results():
    dctx = zstd.ZstdDecompressor()
    with open(SKIM_RESULTS_FILENAME, "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data


def get_egt_data():
    menage_df = pd.read_csv(os.path.join(EGT_DIR, "Menages_semaine.csv"))
    personne_df = pd.read_csv(os.path.join(EGT_DIR, "Personnes_semaine.csv"))
    trip_df = pd.read_csv(os.path.join(EGT_DIR, "Deplacements_semaine.csv"))
    trajet_df = pd.read_csv(os.path.join(EGT_DIR, "Trajets_semaine.csv"))
    df = pd.merge(menage_df, personne_df, on="NQUEST", how="right")
    df = trip_df.merge(df, how="left", on=["NQUEST", "NP"])
    df = trajet_df.merge(df, how="left", on=["NQUEST", "NP", "ND"])
    df["td"] = df["ORH"] * 3600 + df["ORM"] * 60
    df["ta"] = df["DESTH"] * 3600 + df["DESTM"] * 60
    df["tt"] = df["ta"] - df["td"]
    return df


def get_filtered_egt_data(
    egt_data,
    period_start=0,
    period_end=30 * 3600,
    modes=None,
    longest_trip_only=False,
    remove_intrazonal=False,
):
    egt_data = egt_data.loc[egt_data["td"] >= period_start]
    egt_data = egt_data.loc[egt_data["ta"] <= period_end]
    if isinstance(modes, list):
        egt_data = egt_data.loc[egt_data["MOYEN"].isin(modes)]
    if longest_trip_only:
        idx = egt_data.groupby(["NQUEST", "NP"])["tt"].idxmax()
        egt_data = egt_data.loc[idx]
    if remove_intrazonal:
        egt_data = egt_data.loc[egt_data["ORC"] != egt_data["DESTC"]]
    return egt_data.copy()


def get_tt_data():
    google = pd.read_csv(
        os.path.join(TT_DIR, "donne_google.csv"), dtype={"origine": int, "Destination": int}
    )
    google["tt"] = pd.to_timedelta(google["Durée de trajet"]).dt.total_seconds()
    google["td"] = pd.to_datetime(google["temps_depart"], format="%H:%M").dt.time.apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second
    )
    google["date"] = pd.to_datetime(google["date"])
    # Remove weekends.
    google = google.loc[~google["date"].dt.weekday.isin((5, 6))].copy()
    google["source"] = "Google"
    google.rename(columns={"Destination": "destination", "origine": "origin"}, inplace=True)
    google.drop(columns=["Durée de trajet", "temps_depart"], inplace=True)

    tomtom = pd.read_csv(
        os.path.join(TT_DIR, "donne_TomTom.csv"), dtype={"origine": int, "Destination": int}
    )
    tomtom["tt"] = pd.to_timedelta(tomtom["Durée de trajet"]).dt.total_seconds()
    tomtom["td"] = pd.to_datetime(tomtom["temps_depart"], format="%H:%M").dt.time.apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second
    )
    tomtom["date"] = pd.to_datetime(tomtom["date"])
    # Remove weekends.
    tomtom = tomtom.loc[~tomtom["date"].dt.weekday.isin((5, 6))].copy()
    tomtom["source"] = "TomTom"
    tomtom.rename(columns={"Destination": "destination", "origine": "origin"}, inplace=True)
    tomtom.drop(columns=["Durée de trajet", "temps_depart"], inplace=True)

    here = pd.read_csv(
        os.path.join(TT_DIR, "donne_herdeveloper.csv"), dtype={"origine": int, "Destination": int}
    )
    here["tt"] = pd.to_timedelta(here["Durée de trajet"]).dt.total_seconds()
    here["td"] = pd.to_datetime(here["temps_depart"], format="%H:%M").dt.time.apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second
    )
    here["date"] = pd.to_datetime(here["date"])
    # Remove weekends.
    here = here.loc[~here["date"].dt.weekday.isin((5, 6))].copy()
    here["source"] = "HERE"
    here.rename(columns={"Destination": "destination", "origine": "origin"}, inplace=True)
    here.drop(columns=["Durée de trajet", "temps_depart"], inplace=True)

    return pd.concat([google, tomtom, here])


def get_edges():
    edges = gpd.read_file(EDGE_FILENAME)
    # TODO: Remove this when the input of Metropolis is fixed.
    edges["index"] = edges.index
    return edges


def get_edge_with_congestion(edges, weights, bins):
    array = np.empty((len(weights["road_network"][0]), len(bins)))
    for i, w in enumerate(weights["road_network"][0]):
        if isinstance(w, float):
            array[i, :] = w
        else:
            (xs, ys) = np.array([(p["x"], p["y"]) for p in w["points"]]).T
            array[i, :] = np.interp(bins, xs, ys)
    array /= np.atleast_2d(array[:, 0]).T
    edges = edges.copy()
    for i in range(len(bins)):
        col = f"TD{i}"
        edges[col] = array[:, i]
        edges[col] = edges[col].astype(float)
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

    m = folium.Map(
        location=mean_location,
        zoom_start=13,
        tiles="https://api.maptiler.com/maps/basic-v2/256/{z}/{x}/{y}.png?key=ReELeWjLPpebJEd9Ss1D",
        attr='\u003ca href="https://www.maptiler.com/copyright/" target="_blank"\u003e\u0026copy; MapTiler\u003c/a\u003e \u003ca href="https://www.openstreetmap.org/copyright" target="_blank"\u003e\u0026copy; OpenStreetMap contributors\u003c/a\u003e',
    )
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
    for i in range(len(route)):
        leg = route[i]
        edge = edges_taken.loc[leg["edge"]]
        edge_coords = list(edge["geometry"].coords)
        edge_coords = [p[::-1] for p in edge_coords]
        if i + 1 < len(route):
            edge_exit = route[i + 1]["edge_entry"]
        else:
            edge_exit = agent["arrival_time"]
        edge_tt = edge_exit - leg["edge_entry"]
        congestion = edge["fftt"] / edge_tt
        color = to_hex(colormap(congestion))
        tooltip = "From {} to {}<br>Travel time: {}<br>Free-flow tt: {}".format(
            get_time_str(leg["edge_entry"]),
            get_time_str(edge_exit),
            get_tt_str(edge_tt),
            get_tt_str(edge["fftt"]),
        )
        PolyLine(
            locations=edge_coords, tooltip=tooltip, opacity=0.5, color=color, weight=10
        ).add_to(m)
    return m


def get_agent_dataframe(input_agents, agent_results):
    data = [
        {
            "id": a["id"],
            "leg_id": i + 1,
            "utility": l["travel_utility"] + l["schedule_utility"],
            "departure_time": l["departure_time"],
            "arrival_time": l["arrival_time"],
            "expected_utility": a["expected_utility"],
            "expected_arrival_time": l["class"]["value"].get("exp_arrival_time"),
            "road_time": l["class"]["value"].get("road_time"),
            "in_bottleneck_time": l["class"]["value"].get("in_bottleneck_time"),
            "out_bottleneck_time": l["class"]["value"].get("out_bottleneck_time"),
            "nb_edges": len(l["class"]["value"].get("route", [])),
        }
        for a in agent_results
        for i, l in enumerate(a["mode_results"]["value"]["legs"])
    ]
    df = pd.DataFrame(data)
    data = [
        {
            "id": a["id"],
            "leg_id": i + 1,
            "origin": l["class"]["value"].get("origin"),
            "destination": l["class"]["value"].get("destination"),
            "beta": l.get("schedule_utility", {}).get("value", {}).get("beta", 0) * 3600,
            "gamma": l.get("schedule_utility", {}).get("value", {}).get("gamma", 0) * 3600,
            "t_star": (
                l.get("schedule_utility", {}).get("value", {}).get("t_star_low", 0)
                + l.get("schedule_utility", {}).get("value", {}).get("t_star_high", 0)
            )
            / 2,
            "alpha": -l.get("travel_utility", {}).get("value", {}).get("b", 0) * 3600,
            "u": a["modes"][0]["value"]["departure_time_model"]["value"]["choice_model"]["value"][
                "u"
            ],
        }
        for a in input_agents
        for i, l in enumerate(a["modes"][0]["value"]["legs"])
    ]
    df = df.merge(pd.DataFrame(data), on=["id", "leg_id"], how="left")
    df["delay"] = df["arrival_time"] - df["t_star"]
    df["exp_delay"] = df["expected_arrival_time"] - df["t_star"]
    df["delay_cost"] = -(df["beta"] / 3600) * np.minimum(df["delay"], 0.0) + (
        df["gamma"] / 3600
    ) * np.maximum(df["delay"], 0.0)
    df["exp_delay_cost"] = -(df["beta"] / 3600) * np.minimum(df["exp_delay"], 0.0) + (
        df["gamma"] / 3600
    ) * np.maximum(df["exp_delay"], 0.0)
    df["travel_time"] = df["arrival_time"] - df["departure_time"]
    df["travel_cost"] = df["delay_cost"] + df["travel_time"] * df["alpha"] / 3600
    df["exp_travel_time"] = df["expected_arrival_time"] - df["departure_time"]
    df["exp_travel_cost"] = df["exp_delay_cost"] + df["exp_travel_time"] * df["alpha"] / 3600
    return df


def plot_departure_times_cdf(agent_df, egt_data=None):
    fig, ax = plt.subplots()
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["departure_time"] / 3600, egt_data["td"] / 3600],
            bins=300,
            weights=weights,
            density=True,
            cumulative=True,
            histtype="step",
            label=["Metropolis", "EGT"],
        )
        ax.legend()
    else:
        ax.hist(
            agent_df["departure_time"] / 3600,
            bins=300,
            density=True,
            cumulative=True,
            histtype="step",
        )
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Cumulative density")
    fig.tight_layout()
    return fig


def plot_arrival_times_cdf(agent_df, egt_data=None, m=None, M=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["arrival_time"] / 3600, egt_data["ta"] / 3600],
            bins=1000,
            weights=weights,
            density=True,
            cumulative=True,
            histtype="step",
            label=["Metropolis", "EGT"],
        )
        ax.legend(loc="upper left")
    else:
        ax.hist(
            agent_df["arrival_time"] / 3600,
            bins=1000,
            density=True,
            cumulative=True,
            histtype="step",
        )
    ax.set_xlabel("Arrival time (h)")
    ax.set_ylabel("Cumulative density")
    if m is not None:
        ax.set_xlim(left=m)
    if M is not None:
        ax.set_xlim(right=M)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_exp_arrival_times_cdf(agent_df, egt_data=None):
    fig, ax = plt.subplots()
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["expected_arrival_time"] / 3600, egt_data["ta"] / 3600],
            bins=300,
            weights=weights,
            density=True,
            cumulative=True,
            histtype="step",
            label=["Metropolis", "EGT"],
        )
        ax.legend()
    else:
        ax.hist(
            agent_df["expected_arrival_time"] / 3600,
            bins=300,
            density=True,
            cumulative=True,
            histtype="step",
        )
    ax.set_xlabel("Expected arrival time (h)")
    ax.set_ylabel("Cumulative density")
    fig.tight_layout()
    return fig


def plot_desired_arrival_times_cdf(agent_df):
    fig, ax = plt.subplots()
    ax.hist(
        agent_df["t_star"] / 3600,
        bins=300,
        density=True,
        cumulative=True,
        histtype="step",
    )
    ax.set_xlabel("Desired arrival time (h)")
    ax.set_ylabel("Cumulative density")
    fig.tight_layout()
    return fig


def plot_times_cdf(agent_df):
    fig, ax = plt.subplots()
    ax.hist(
        agent_df[["departure_time", "arrival_time", "expected_arrival_time", "t_star"]] / 3600,
        bins=300,
        density=True,
        cumulative=True,
        histtype="step",
        label=["Dep. time", "Arr. time", "Exp. arr. time", "Desired arr. time"],
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cumulative density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_departure_times_hist(agent_df, bins=None, egt_data=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["departure_time"] / 3600, egt_data["td"] / 3600],
            bins=bins,
            weights=weights,
            density=True,
            label=["Metropolis", "EGT"],
        )
        ax.legend()
    else:
        ax.hist(agent_df["departure_time"] / 3600, bins=bins)
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_arrival_times_hist(agent_df, bins=None, egt_data=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["arrival_time"] / 3600, egt_data["ta"] / 3600],
            bins=bins,
            weights=weights,
            density=True,
            label=["Metropolis", "EGT"],
        )
        ax.legend()
    else:
        ax.hist(agent_df["arrival_time"] / 3600, bins=bins)
    ax.set_xlabel("Arrival time (h)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_expected_arrival_times_hist(agent_df, bins=None, egt_data=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["expected_arrival_time"] / 3600, egt_data["ta"] / 3600],
            bins=bins,
            weights=weights,
            density=True,
            label=["Metropolis", "EGT"],
        )
        ax.legend()
    else:
        ax.hist(agent_df["expected_arrival_time"] / 3600, bins=bins)
    ax.set_xlabel("Expected arrival time (h)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_desired_arrival_times_hist(agent_df, bins=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    ax.hist(agent_df["t_star"] / 3600, bins=bins)
    ax.set_xlabel("Desired arrival time (h)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_times_hist(agent_df, bins=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    ax.hist(
        agent_df[["departure_time", "arrival_time", "expected_arrival_time", "t_star"]] / 3600,
        bins=bins,
        label=["Dep. time", "Arr. time", "Exp. arr. time", "Desired arr. time"],
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_delay_hist(agent_df, bins=None, cumulative=False, bounds=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    if bounds is not None:
        agent_df = agent_df.loc[
            (agent_df["delay"] / 60 > bounds[0]) & (agent_df["delay"] / 60 <= bounds[1])
        ]
    if cumulative:
        ax.hist(
            agent_df[["delay", "exp_delay"]] / 60,
            bins=300,
            density=True,
            cumulative=True,
            histtype="step",
            label=["Actual delay", "Expected delay"],
        )
        ax.grid()
        ax.legend()
    else:
        ax.hist(
            agent_df[["delay", "exp_delay"]] / 60,
            bins=bins,
            label=["Actual delay", "Expected delay"],
        )
    ax.set_xlabel("Delay (min)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_delay_cost_hist(agent_df, bins=None):
    fig, ax = plt.subplots()
    if bins is None:
        bins = 60
    ax.hist(
        agent_df[["delay_cost", "exp_delay_cost"]],
        bins=bins,
        label=["Actual delay cost", "Expected delay cost"],
    )
    ax.set_xlabel("Delay cost (€)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_delay_u_scatter(agent_df):
    fig, ax = plt.subplots()
    ax.scatter(agent_df["exp_delay"] / 60, agent_df["u"], alpha=0.01)
    ax.set_xlabel("Expected delay (min)")
    ax.set_ylabel("Dep. time random value")
    fig.tight_layout()
    return fig


def plot_delay_cost_u_scatter(agent_df):
    fig, ax = plt.subplots()
    ax.scatter(agent_df["exp_delay_cost"], agent_df["u"], alpha=0.01)
    ax.set_xlabel("Expected delay cost (€)")
    ax.set_ylabel("Dep. time random value")
    fig.tight_layout()
    return fig


def plot_travel_cost_u_scatter(agent_df):
    fig, ax = plt.subplots()
    ax.scatter(agent_df["exp_travel_cost"], agent_df["u"], alpha=0.01)
    ax.set_xlabel("Expected travel cost (€)")
    ax.set_ylabel("Dep. time random value")
    fig.tight_layout()
    return fig


def plot_exp_travel_time_scatter(agent_df, bound=0, bound_per=0, M=None):
    fig, ax = plt.subplots()
    if M is None:
        M = max(agent_df["exp_travel_time"].max(), agent_df["travel_time"].max()) / 60
    ax.scatter(agent_df["exp_travel_time"] / 60, agent_df["travel_time"] / 60, alpha=0.01)
    ax.plot([0, M], [0, M], color="black")
    if bound:
        ax.plot([bound, M + bound], [0, M], color="black", linestyle="dashed")
        ax.plot([0, M], [bound, M + bound], color="black", linestyle="dashed")
    if bound_per:
        ax.plot([0, M * bound_per], [0, M], color="black", linestyle="dashed")
        ax.plot([0, M], [0, M * bound_per], color="black", linestyle="dashed")
    ax.set_xlabel("Expected travel time (min)")
    ax.set_ylabel("Actual travel time (min)")
    ax.set_xlim(0, M)
    ax.set_ylim(0, M)
    fig.tight_layout()
    return fig


def plot_travel_time_hist(agent_df, egt_data=None, bins=None, M=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if M is None:
        M = agent_df["travel_time"].max() / 60
    if bins is None:
        bins = 60
    if egt_data is not None:
        weights = [np.repeat(1, len(agent_df)), egt_data["POIDSP"]]
        ax.hist(
            [agent_df["travel_time"] / 60, egt_data["tt"] / 60],
            bins=(M or 100) * 10,
            weights=weights,
            density=True,
            cumulative=True,
            histtype="step",
            label=["Metropolis", "EGT"],
        )
        ax.legend(loc="lower right")
        ax.grid()
        ax.set_ylabel("Cumulative density")
        ax.set_ylim(0, 1)
    else:
        ax.hist(agent_df["travel_time"] / 60, bins=bins)
        ax.set_ylabel("Count")
    ax.set_xlabel("Travel time (min)")
    ax.set_xlim(0, M)
    fig.tight_layout()
    return fig


def plot_weight_travel_time_function(edge_id, weights):
    ttf = weights["road_network"][0][edge_id]
    xs = np.array([p["x"] for p in ttf["points"]])
    ys = np.array([p["y"] for p in ttf["points"]])
    m = ttf["period"][0]
    M = ttf["period"][1]
    fig, ax = plt.subplots()
    ax.plot(xs / 3600, ys, "-o", markersize=1.5, alpha=0.7)
    ax.set_xlim(m / 3600, M / 3600)
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Travel time (s)")
    fig.tight_layout()
    return fig


def plot_od_travel_time_function(origin, destination, skims):
    ttf = skims["road_network"][0]["profile_query_cache"][str(origin)][str(destination)]
    xs = np.array([p["x"] for p in ttf["points"]])
    ys = np.array([p["y"] for p in ttf["points"]])
    m = ttf["period"][0]
    M = ttf["period"][1]
    fig, ax = plt.subplots()
    ax.plot(xs / 3600, ys / 60, "-o", markersize=1.5, alpha=0.7)
    ax.set_xlim(m / 3600, M / 3600)
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Travel time (min.)")
    fig.tight_layout()
    return fig


def plot_travel_time_comparison(origin, destination, skims, tt_data, od_pairs):
    ttf = skims["road_network"][0]["profile_query_cache"][str(origin)][str(destination)]
    xs = np.array([p["x"] for p in ttf["points"]])
    ys = np.array([p["y"] for p in ttf["points"]])
    m = ttf["period"][0]
    M = ttf["period"][1]

    origin_id = od_pairs.loc[od_pairs["origin_id"] == origin, "zone_origin"].iloc[0]
    destination_id = od_pairs.loc[
        od_pairs["destination_id"] == destination, "zone_destination"
    ].iloc[0]
    tt_data = tt_data.loc[
        (tt_data["origin"] == origin_id) & (tt_data["destination"] == destination_id)
    ]
    tt_data = tt_data.loc[(tt_data["td"] >= m) & (tt_data["td"] <= M)]

    google_tts = tt_data.loc[tt_data["source"] == "Google"].groupby("td")["tt"].describe()
    tomtom_tts = tt_data.loc[tt_data["source"] == "TomTom"].groupby("td")["tt"].describe()
    here_tts = tt_data.loc[tt_data["source"] == "HERE"].groupby("td")["tt"].describe()

    fig, ax = plt.subplots()
    ax.plot(xs / 3600, ys / 60, "-o", markersize=2, alpha=0.5, color="green", label="Metropolis")
    ax.plot(
        google_tts.index / 3600,
        google_tts["50%"] / 60,
        "-o",
        markersize=2,
        alpha=0.5,
        color="orange",
        label="Google",
    )
    ax.plot(google_tts.index / 3600, google_tts["min"] / 60, "--", alpha=0.5, color="orange")
    ax.plot(google_tts.index / 3600, google_tts["max"] / 60, "--", alpha=0.5, color="orange")
    ax.plot(
        tomtom_tts.index / 3600,
        tomtom_tts["50%"] / 60,
        "-o",
        markersize=2,
        alpha=0.5,
        color="red",
        label="TomTom",
    )
    ax.plot(tomtom_tts.index / 3600, tomtom_tts["min"] / 60, "--", alpha=0.5, color="red")
    ax.plot(tomtom_tts.index / 3600, tomtom_tts["max"] / 60, "--", alpha=0.5, color="red")
    ax.plot(
        here_tts.index / 3600,
        here_tts["50%"] / 60,
        "-o",
        markersize=2,
        alpha=0.5,
        color="blue",
        label="HERE",
    )
    ax.plot(here_tts.index / 3600, here_tts["min"] / 60, "--", alpha=0.5, color="blue")
    ax.plot(here_tts.index / 3600, here_tts["max"] / 60, "--", alpha=0.5, color="blue")
    ax.set_xlim(m / 3600, M / 3600)
    ax.set_xlabel("Departure time (h)")
    ax.set_ylabel("Travel time (min.)")
    ax.legend()
    fig.tight_layout()
    return fig


def get_chevelus(origin, destination, agent_df, agent_results, edges):
    agent_ids = agent_df.loc[
        (agent_df["origin"] == origin) & (agent_df["destination"] == destination), "id"
    ]
    edge_counts = defaultdict(lambda: 0)
    for i in agent_ids:
        route = agent_results[i]["mode_results"]["value"]["route"]
        for e in route:
            edge_counts[e["edge"]] += 1
    edges = edges.loc[edges["index"].isin(edge_counts)].copy()
    edges["count"] = pd.Series(edge_counts)
    return edges


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

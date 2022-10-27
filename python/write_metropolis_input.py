import os
import json

#  import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the directory where the node and edge files are stored.
ROAD_NETWORK_DIR = "./output/here_network/"
# Path to the file where the trip data is stored.
TRIPS_FILE = "./output/trips.csv"
# Path to the directory where the simulation input should be stored.
OUTPUT_DIR = "./output/metropolis_input/"
# Vehicle length in meters.
VEHICLE_LENGTH = 10.0 * 10.0
# Vehicle passenger-car equivalent.
VEHICLE_PCE = 10.0 * 1.0
# Period in which the departure time of the trip is chosen.
PERIOD = [6.0 * 3600.0, 12.0 * 3600.0]

print("Reading edges")
edges = gpd.read_file(os.path.join(ROAD_NETWORK_DIR, "edges.fgb"))

print("Creating Metropolis road network")
metro_edges = list()
for _, row in edges.iterrows():
    edge = [
        row["source_index"],
        row["target_index"],
        {
            "id": int(row["LINK_ID"]),
            "base_speed": float(row["speed"]) / 3.6,
            "length": float(row["length"]),
            "lanes": int(row["lanes"]),
            "speed_density": {
                "type": "FreeFlow",
            },
        },
    ]
    metro_edges.append(edge)

graph = {
    "edges": metro_edges,
}

vehicles = [
    {
        "length": VEHICLE_LENGTH,
        "pce": VEHICLE_PCE,
        "speed_function": {
            "type": "Base",
        }
    }
]

road_network = {
    "graph": graph,
    "vehicles": vehicles,
}

print("Reading trips")
trips = pd.read_csv(TRIPS_FILE)

print("Generating agents")
agents = list()
for key, row in trips.iterrows():
    alpha = 10.0
    beta = 5.0
    gamma = 20.0
    delta = 0.0
    t_star = row["arrival_time"]
    departure_time_model = {
        "type": "Constant",
        "values": row["departure_time"],
    }
    car_mode = {
        "type": "Road",
        "values": {
            "origin": row["origin_id"],
            "destination": row["destination_id"],
            "vehicle": 0,
            "utility_model": {
                "type": "Proportional",
                "values": -alpha,
            },
            "departure_time_model": departure_time_model,
        },
    }
    agent = {
        "id": key,
        "schedule_utility": {
            "type": "AlphaBetaGamma",
            "values": {
                "beta": beta,
                "gamma": gamma,
                "t_star_high": t_star + delta / 2.0,
                "t_star_low": t_star - delta / 2.0,
            },
        },
        "modes": [car_mode],
    }
    agents.append(agent)

print("Writing data...")
with open(os.path.join(OUTPUT_DIR, "network.json"), "w") as f:
    f.write(json.dumps(road_network))

with open(os.path.join(OUTPUT_DIR, "agents.json"), "w") as f:
    f.write(json.dumps(agents))

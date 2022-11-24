import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the directory where the node and edge files are stored.
ROAD_NETWORK_DIR = "./output/here_network/"
# Path to the file where the trip data is stored.
TRIPS_FILE = "./output/trips_filtered.csv"
# Path to the directory where the simulation input should be stored.
OUTPUT_DIR = "./output/server_runs/next_run/"
# Vehicle length in meters.
VEHICLE_LENGTH = 15.0 * 10.0
# Vehicle passenger-car equivalent.
VEHICLE_PCE = 15.0 * 1.0
# Period in which the departure time of the trip is chosen.
PERIOD = [3.0 * 3600.0, 10.0 * 3600.0]
# Capacity of the different edge road types.
CAPACITY = {
    0: None,  # Infinite capacity.
    1: 2000,
    2: 1500,
    3: 800,
    4: 600,
}
# If True, enable entry bottleneck using capacity defined by `CAPACITY`.
ENTRY_BOTTLENECK = True
# If True, enable exit bottleneck using capacity defined by `CAPACITY`.
EXIT_BOTTLENECK = False
# Value of time in the car, in euros / hour.
ALPHA = 10.0
# Value of arriving early at destination, in euros / hour.
BETA = 5.0
# Value of arriving late at destination, in euros / hour.
GAMMA = 20.0
# Time window for on-time arrival, in seconds.
DELTA = 0.0
# If True, departure time is endogenous.
ENDOGENOUS_DEPARTURE_TIME = True
# Value of μ for the departure-time model (if ENDOGENOUS_DEPARTURE_TIME is True).
DT_MU = 2.0

# Parameters to use for the simulation.
PARAMETERS = {
    "period": PERIOD,
    "learning_model": {
        "type": "Exponential",
        "value": {
            "alpha": 0.99,
        },
    },
    "stopping_criteria": [
        {
            "type": "MaxIteration",
            "value": 100,
        },
    ],
    "update_ratio": 1.0,
    "random_seed": 13081996,
    "network": {
        "road_network": {
            "recording_interval": 300.0,
            "simulated_simplification": {
                "type": "Bound",
                "value": 2.0,
            },
            "weight_simplification": {
                "type": "Bound",
                "value": 2.0,
            },
            "overlay_simplification": {
                "type": "Bound",
                "value": 2.0,
            },
            "search_space_simplification": {
                "type": "Bound",
                "value": 1.0,
            },
        }
    },
}

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
    if capacity := CAPACITY.get(row["road_type"]):
        if ENTRY_BOTTLENECK:
            edge[2]["bottleneck_inflow"] = capacity / 3600.0
        if EXIT_BOTTLENECK:
            edge[2]["bottleneck_outflow"] = capacity / 3600.0
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
        },
    }
]

road_network = {
    "graph": graph,
    "vehicles": vehicles,
}

print("Reading trips")
trips = pd.read_csv(TRIPS_FILE)

print("Generating agents")
random_u = iter(np.random.uniform(size=len(trips)))
agents = list()
for key, row in trips.iterrows():
    t_star = row["arrival_time"]
    if ENDOGENOUS_DEPARTURE_TIME:
        departure_time_model = {
            "type": "ContinuousChoice",
            "value": {
                "period": PERIOD,
                "choice_model": {
                    "type": "Logit",
                    "value": {
                        "u": next(random_u),
                        "mu": DT_MU,
                    },
                },
            },
        }
    else:
        departure_time_model = {
            "type": "Constant",
            "value": row["departure_time"],
        }
    car_mode = {
        "type": "Road",
        "value": {
            "origin": row["origin_id"],
            "destination": row["destination_id"],
            "vehicle": 0,
            "utility_model": {
                "type": "Proportional",
                "value": -ALPHA / 3600.0,
            },
            "departure_time_model": departure_time_model,
        },
    }
    agent = {
        "id": key,
        "schedule_utility": {
            "type": "AlphaBetaGamma",
            "value": {
                "beta": BETA / 3600.0,
                "gamma": GAMMA / 3600.0,
                "t_star_high": t_star + DELTA / 2.0,
                "t_star_low": t_star - DELTA / 2.0,
                "desired_arrival": True,
            },
        },
        "modes": [car_mode],
    }
    agents.append(agent)

print("Writing data...")
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "network.json"), "w") as f:
    f.write(json.dumps(road_network))
with open(os.path.join(OUTPUT_DIR, "agents.json"), "w") as f:
    f.write(json.dumps(agents))
with open(os.path.join(OUTPUT_DIR, "parameters.json"), "w") as f:
    f.write(json.dumps(PARAMETERS))

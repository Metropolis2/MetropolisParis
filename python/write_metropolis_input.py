import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the directory where the node and edge files are stored.
ROAD_NETWORK_DIR = "./output/osm_network/"
# Path to the file where the trip data is stored.
TRIPS_FILE = "./output/trips_filtered.csv"
# Path to the directory where the simulation input should be stored.
OUTPUT_DIR = "./output/next_run/"
# Vehicle length in meters.
VEHICLE_LENGTH = 10.0 * 10.0
# Vehicle passenger-car equivalent.
VEHICLE_PCE = 10.0 * 1.0
# Period in which the departure time of the trip is chosen.
PERIOD = [3.0 * 3600.0, 10.0 * 3600.0]
# Capacity of the different edge road types.
CAPACITY = {
    0: None,
    1: 2000,
    2: 2000,
    3: 1500,
    4: 800,
    5: 600,
    6: 600,
    7: 600,
    8: 1500,
    9: 1500,
    10: 1500,
    11: 800,
    12: 600,
    13: 300,
    14: 300,
    15: 300,
}
# If True, enable entry bottleneck using capacity defined by `CAPACITY`.
ENTRY_BOTTLENECK = True
# If True, enable exit bottleneck using capacity defined by `CAPACITY`.
EXIT_BOTTLENECK = False
# Value of time in the car, in euros / hour.
ALPHA = 15.0
# Value of arriving early at destination, in euros / hour.
BETA = 7.5
# Value of arriving late at destination, in euros / hour.
GAMMA = 30.0
# Time window for on-time arrival, in seconds.
DELTA = 0.0
# If True, departure time is endogenous.
ENDOGENOUS_DEPARTURE_TIME = True
# Value of μ for the departure-time model (if ENDOGENOUS_DEPARTURE_TIME is True).
DT_MU = 3.0
# How t* is computed given the observed arrival time.
def T_STAR_FUNC(ta):
    return ta
CONST_TT = {
    3: 10,
    4: 15,
    5: 20,
    6: 25,
    7: 30,
    8: 35,
    9: 40,
    10: 45,
}
# Seed for the random number generators.
SEED = 13081996
RNG = np.random.default_rng(SEED)

# Parameters to use for the simulation.
PARAMETERS = {
    "period": PERIOD,
    "init_iteration_counter": 1,
    "learning_model": {
        "type": "Exponential",
        "value": {
            "alpha": 0.99,
        },
    },
    "stopping_criteria": [
        {
            "type": "MaxIteration",
            "value": 5,
        },
    ],
    "update_ratio": 1.0,
    "random_seed": SEED,
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

print("Creating Metropolis road network")
metro_edges = list()
for _, row in edges.iterrows():
    edge = [
        row["source_index"],
        row["target_index"],
        {
            "id": int(row["index"]),
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
    if const_tt := CONST_TT.get(row["neighbor_count"]):
        edge[2]["constant_travel_time"] = const_tt
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
random_u = iter(RNG.uniform(size=len(trips)))
agents = list()
for person_id, idx in trips.groupby('person_id').groups.items():
    legs = list()
    prev_ta = None
    for key, trip in trips.loc[idx].iterrows():
        t_star = T_STAR_FUNC(trip["arrival_time"])
        leg = {
            "travel_utility": {
                "type": "Polynomial",
                "value": {
                    "b": -ALPHA / 3600.0,
                }
            },
            "schedule_utility": {
                "type": "AlphaBetaGamma",
                "value": {
                    "beta": BETA / 3600.0,
                    "gamma": GAMMA / 3600.0,
                    "t_star_high": t_star + DELTA / 2.0,
                    "t_star_low": t_star - DELTA / 2.0,
                },
            },
            "class": {
                "type": "Road",
                "value": {
                    "origin": trip['origin_id'],
                    "destination": trip['destination_id'],
                    "vehicle": 0,
                },
            },
        }
        if not prev_ta is None:
            # Set stopping time of previous leg.
            legs[-1]["stopping_time"] = trip['departure_time'] - prev_ta
        prev_ta = trip['arrival_time']
        legs.append(leg)
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
            "value": trips.loc[idx[0], "departure_time"],
        }
    car_mode = {
        "type": "Trip",
        "value": {
            "legs": legs,
            "departure_time_model": departure_time_model,
        },
    }
    agent = {
        "id": person_id,
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

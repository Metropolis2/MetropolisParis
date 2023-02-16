import numpy as np
import pandas as pd

# Path to the file where the trips are stored.
TRIP_FILENAME = "./output/trips_full.csv"
# Path to the file where the mapping between IRIS zone ids and node ids is stored.
ZONE_ID_FILE = "./output/zone_id_map_osm.csv"
# Output file of the generated trips.
OUTPUT_FILENAME = "./output/trips_filtered.csv"
# Returns only trips whose departure time is later than this value (in seconds after midnight).
START_TIME = 3.0 * 3600.0
# Returns only trips whose arrival time is earlier than this value (in seconds after midnight).
END_TIME = 10.0 * 3600.0
# List of modes used to filter trips.
MODE = ["car"]
# If True, remove all intra-zonal trips.
REMOVE_INTRA_ZONAL = True
# If True, only keep the longest trip of each individual.
ONE_TRIP_MAX = False
# Chunk size used when reading the input CSV file, i.e., number of lines read in memory.
# Set to `None` to read all the file in memory.
CHUNK_SIZE = 100_000
# Share of trips that must be randomly selected.
# Set to 1.0 to select all the trips.
SHARE = 0.1
# Set of random seed to repeat the same draws.
RANDOM_SEED = 13081996

rng = np.random.default_rng(RANDOM_SEED)
zone_mapping = pd.read_csv(ZONE_ID_FILE)

df = pd.DataFrame()
tot_tt_before = 0.0
tt_intrazonal = 0.0
nb_intrazonal = 0
tt_onetrip = 0.0
nb_onetrip = 0
nb_trips = 0
person_ids = set()
all_person_ids = set()
with pd.read_csv(TRIP_FILENAME, chunksize=CHUNK_SIZE) as reader:
    for chunk in reader:
        # Filter by start time, end time and mode.
        chunk = chunk.loc[
            (chunk["departure_time"] >= START_TIME)
            & (chunk["arrival_time"] <= END_TIME)
            & (chunk["mode"].isin(list(MODE)))
        ].copy()
        tot_tt_before += chunk["travel_time"].sum()
        if REMOVE_INTRA_ZONAL:
            mask = chunk["iris_origin"] == chunk["iris_destination"]
            tt_intrazonal += chunk.loc[mask, "travel_time"].sum()
            nb_intrazonal += mask.sum()
            chunk = chunk.loc[~mask].copy()
        if ONE_TRIP_MAX:
            # Keeping the longest trip of each individual.
            indices = chunk.groupby("person_id")["travel_time"].idxmax()
            tt_onetrip += chunk["travel_time"].sum() - chunk.loc[indices, "travel_time"].sum()
            nb_onetrip += len(chunk) - len(indices)
            chunk = chunk.loc[indices].copy()
        nb_trips += len(chunk)
        # Randomly select individuals.
        if SHARE < 1.0:
            if SHARE <= 0.0:
                print("`SHARE` must be positive, got {}".format(SHARE))
                import sys

                sys.exit()
            # A person can be split in two chunks so we have to make sure that we keep all of his
            # trips.
            new_persons = set(chunk["person_id"].unique())  # Person ids in the chunk.
            new_persons = new_persons.difference(
                all_person_ids
            )  # Person ids not encountered before.
            all_person_ids = all_person_ids.union(new_persons)
            new_persons = rng.choice(
                np.array(list(new_persons)), size=round(SHARE * len(new_persons)), replace=False
            )
            person_ids = person_ids.union(new_persons)
            chunk = chunk.loc[chunk["person_id"].isin(person_ids)].copy()
        # Set source and target node id of the trips.
        chunk.drop(columns=['origin_id', 'destination_id'], errors='ignore', inplace=True)
        chunk = (
            chunk.merge(
                zone_mapping.loc[zone_mapping["in"], ["node_id", "CODE_IRIS"]],
                left_on="iris_destination",
                right_on="CODE_IRIS",
            )
            .rename(columns={"node_id": "destination_id"})
            .drop(columns="CODE_IRIS")
        )
        chunk = (
            chunk.merge(
                zone_mapping.loc[~zone_mapping["in"], ["node_id", "CODE_IRIS"]],
                left_on="iris_origin",
                right_on="CODE_IRIS",
            )
            .rename(columns={"node_id": "origin_id"})
            .drop(columns="CODE_IRIS")
        )
        df = pd.concat((df, chunk))

if nb_intrazonal > 0:
    print(
        "Removed {} intra-zonal trips, representing {:.2f} % of total travel time".format(
            nb_intrazonal, 100 * tt_intrazonal / tot_tt_before
        )
    )

if nb_onetrip > 0:
    print(
        "Removed {} duplicate-agent trips, representing {:.2f} % of total travel time".format(
            nb_onetrip, 100 * tt_onetrip / tot_tt_before
        )
    )

if SHARE < 1.0:
    print("Total number of valid trips: {}".format(nb_trips))
    print("Total number of trips randomly selected: {}".format(len(df)))

print("Writing data")
df.sort_values(["person_id", "trip_index"], inplace=True)
df.to_csv(OUTPUT_FILENAME, index=False)

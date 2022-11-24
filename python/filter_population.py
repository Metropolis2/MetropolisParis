import pandas as pd

# Path to the file where the trips are stored.
TRIP_FILENAME = "/home/ljavaudin/Projects/MetropolisIDF/output/trips.csv"
# Output file of the generated trips.
OUTPUT_FILENAME = "/home/ljavaudin/Projects/MetropolisIDF/output/trips_filtered.csv"
# Returns only trips whose departure time is later than this value (in seconds after midnight).
START_TIME = 3.0 * 3600.0
# Returns only trips whose arrival time is earlier than this value (in seconds after midnight).
END_TIME = 10.0 * 3600.0
# List of modes used to filter trips.
MODE = ["car"]
# If True, remove all intra-zonal trips.
REMOVE_INTRA_ZONAL = True
# If True, only keep the longest trip of each individual.
ONE_TRIP_MAX = True


df = pd.read_csv(TRIP_FILENAME)
# Filter by start time, end time and mode.
df = df.loc[
    (df["departure_time"] >= START_TIME)
    & (df["arrival_time"] <= END_TIME)
    & (df["mode"].isin(MODE))
].copy()
if REMOVE_INTRA_ZONAL:
    tot_tt = df["travel_time"].sum()
    mask = df["iris_origin"] == df["iris_destination"]
    df = df.loc[~mask].copy()
    tot_tt2 = df["travel_time"].sum()
    print(
        "Removing {} intra-zonal trips, representing {:.2f} % of total travel time".format(
            mask.sum(), 100 * (tot_tt - tot_tt2) / tot_tt
        )
    )
if ONE_TRIP_MAX:
    # Keeping the longest trip of each individual.
    print("Keeping only the longest trip of each individual...")
    tot_tt = df["travel_time"].sum()
    indices = df.groupby("person_id")["travel_time"].idxmax()
    df = df.loc[indices].copy()
    tot_tt2 = df["travel_time"].sum()
    print("{:.2f} % total travel time removed".format(100 * (tot_tt - tot_tt2) / tot_tt))
print("Writing data")
df.to_csv(OUTPUT_FILENAME, index=False)

import pandas as pd
import numpy as np


#################
### FUNCTIONS ###
#################


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    ----------------
    lon1 : longitude of the first point (float)
    lat1 : latitude of the first point (float)
    lon2 : longitude of the first point (float)
    lat2 : latitude of the first point (float)
    All args must be of equal length.
    ---------------------
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def geo_position(x_bari, y_bari, x, y):
    """
    This function gives for a point with coordinates (x,y)
    its position geographic position (n,w,s,e) relative to
    a barycenter with coordinates (x_bari,y_bari)
    ----------
    x : longitude of the point (float)
    y : latitude of the point (float)
    x_bari : longitude of the barycenter (float)
    y_bari : latitude of the barycenter (float)
    ----------
    string
    n: North
    s: South
    e: East
    w: West
    """
    max_y = max(y_bari, y) + 1
    max_x = max(x_bari, x) + 1
    min_y = min(y_bari, y) - 1
    min_x = min(x_bari, x) - 1
    if (
        y <= ((max_y - min_y) / (min_x - max_x)) * (x - x_bari) + y_bari
        and y > ((min_y - max_y) / (min_x - max_x)) * (x - x_bari) + y_bari
    ):
        return "n"

    if (
        y >= ((max_y - min_y) / (min_x - max_x)) * (x - x_bari) + y_bari
        and y > ((min_y - max_y) / (min_x - max_x)) * (x - x_bari) + y_bari
    ):
        return "e"

    if (
        y >= ((max_y - min_y) / (min_x - max_x)) * (x - x_bari) + y_bari
        and y < ((min_y - max_y) / (min_x - max_x)) * (x - x_bari) + y_bari
    ):
        return "s"

    if (
        y <= ((max_y - min_y) / (min_x - max_x)) * (x - x_bari) + y_bari
        and y < ((min_y - max_y) / (min_x - max_x)) * (x - x_bari) + y_bari
    ):
        return "w"


################
###  INPUTS  ###
################

intersections_path = ".../.csv"
zones_path = ".../.csv"
links_path = ".../.csv"
output_path = ".../links_connectors.csv"

max_speed_connection = 129
min_capacity_connection = 1001
among_closest = 15
connectors_parameters = {"function": 1, "lanes": 5, "speed": 200, "capacity": 99999}


##############
### SCRIPT ###
##############

print("Loading and preparing files...")

# Load files
df_intersections = pd.read_csv(intersections_path)
df_zones = pd.read_csv(zones_path)
df_links = pd.read_csv(links_path)


# Restrict available link to those with:
# - A capacity higher than min_capacity_connection
df_links_ = df_links[(df_links["capacity"] > min_capacity_connection)]
# - A speed lesser than max_speed_connection
df_links_ = df_links[(df_links["speed"] < max_speed_connection)]

print("Compute capacity for each intersection...")

# compute the out/in road capacity
for index, row in df_intersections.iterrows():

    # Compute the out-capacity for each link
    df = df_links_[df_links_["origin"] == row["id"]]
    out_capacity = (df["capacity"] * df["lanes"]).sum()

    # Compute the in-capacity for each link
    df = df_links_[df_links_["destination"] == row["id"]]
    in_capacity = (df["capacity"] * df["lanes"]).sum()

    # Write the in/out-capacity of each intersection in the intersection file
    df_intersections.loc[index, "out_capacity"] = out_capacity
    df_intersections.loc[index, "in_capacity"] = in_capacity


# Create a dataframe of distance between each intersection and each zones
print("Create the connectors...")

i = 1000000
n = len(df_zones["id"])
n_i = 0

for _, r in df_zones.iterrows():
    df_intersections_ = df_intersections.copy()
    df_intersections_ = df_intersections_[df_intersections_["in_capacity"] != 0]
    # Compute the distance between the center and each intersections
    df_intersections_["dist"] = haversine_np(
        r["x"], r["y"], df_intersections_["x"], df_intersections_["y"]
    )
    df_intersections_ = df_intersections_.sort_values(by="dist")
    df_intersections_ = df_intersections_.head(among_closest)

    # Compute the polar position between the center and each point
    df_intersections_["4_zones"] = df_intersections_.apply(
        lambda x: geo_position(r["x"], r["y"], x["x"], x["y"]), axis=1
    )
    # Create connectors for each polar position
    groups = df_intersections_.groupby(["4_zones"])

    for g, df in groups:
        # Create out connectors
        df_ = df[df["in_capacity"] == df["in_capacity"].max()]
        df_ = df_.head(1)
        df_links = df_links.append(
            {
                "id": i,
                "name": "connector from " + str(r["id"]),
                "function": connectors_parameters["function"],
                "lanes": connectors_parameters["lanes"],
                "speed": connectors_parameters["speed"],
                "length": float(df_["dist"].mean()),
                "capacity": connectors_parameters["capacity"],
                "origin": int(df_["id"]),
                "destination": int(r["id"]),
                "osm_id": "",
            },
            ignore_index=True,
        )
        i += 1

for _, r in df_zones.iterrows():
    df_intersections_ = df_intersections.copy()
    df_intersections = df_intersections[df_intersections["in_capacity"] != 0]

    # Compute the distance between the center and each intersections
    df_intersections_["dist"] = haversine_np(
        r["x"], r["y"], df_intersections_["x"], df_intersections_["y"]
    )
    df_intersections_ = df_intersections_.sort_values(by="dist")
    df_intersections_ = df_intersections_.head(among_closest)

    # Compute the polar position between the center and each point
    df_intersections_["4_zones"] = df_intersections_.apply(
        lambda x: geo_position(r["x"], r["y"], x["x"], x["y"]), axis=1
    )
    # Create connectors for each polar position
    groups = df_intersections_.groupby(["4_zones"])

    for g, df in groups:
        df_ = df[df["out_capacity"] == df["out_capacity"].max()]
        df_ = df_.head(1)
        # Create in connectors
        df_links = df_links.append(
            {
                "id": i,
                "name": "connector to " + str(float(r["id"])),
                "function": connectors_parameters["function"],
                "lanes": connectors_parameters["lanes"],
                "speed": connectors_parameters["speed"],
                "length": float(df_["dist"].mean()),
                "capacity": connectors_parameters["capacity"],
                "origin": int(r["id"]),
                "destination": int(df_["id"].mean()),
                "osm_id": "",
            },
            ignore_index=True,
        )
        i += 1

    n_i += 1
    print(str(n_i) + "/" + str(n))

###############
### OUTPUTS ###
###############

df_links.to_csv(output_path)

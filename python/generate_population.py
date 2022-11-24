import os

import shapely.prepared
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

# Path to the directory where the synthetic population files from Eqasim are stored.
EQASIM_OUTPUT = "/home/ljavaudin/Projects/MetropolisIDF/data/synthetic_population/"
# Path to the file where the mapping between IRIS zone ids and node ids is stored.
ZONE_ID_FILE = "./output/zone_id_map.csv"
# Path to the IRIS Shapefile.
IRIS_FILE = "/home/ljavaudin/Projects/MetropolisIDF/data/contours_iris_france/"
# Départements in the studied area, used to filter IRIS zones.
IDF_DEP = ("75", "77", "78", "91", "92", "93", "94", "95")
# Output file of the generated trips.
OUTPUT_FILENAME = "/home/ljavaudin/Projects/MetropolisIDF/output/trips.csv"
# List of modes used to filter trips.
MODE = ["car"]


def get_households():
    """Returns a pd.DataFrame with a description of the households in the population."""
    print("Reading synthetic population households")
    return pd.read_csv(
        os.path.join(EQASIM_OUTPUT, "ile_de_france_households.csv"),
        sep=";",
        usecols=["household_id", "income"],
    )


def get_persons():
    """Returns a pd.DataFrame with a description of the persons in the population."""
    print("Reading synthetic population persons")
    return pd.read_csv(
        os.path.join(EQASIM_OUTPUT, "ile_de_france_persons.csv"),
        sep=";",
        usecols=[
            "person_id",
            "household_id",
            "age",
            "employed",
            "sex",
            "socioprofessional_class",
            "has_driving_license",
            #  "has_pt_subscription",
        ],
    )


def get_zone_mapping():
    return pd.read_csv(ZONE_ID_FILE)


def get_trips():
    """Returns a gpd.GeoDataFrame with the list of trips, for the filtered modes and time."""
    print("Reading synthetic population trips")
    gdf = gpd.read_file(os.path.join(EQASIM_OUTPUT, "ile_de_france_trips.gpkg"))
    # Filter trips with a valid mode.
    gdf = gdf.loc[gdf["mode"].isin(MODE)].copy()
    gdf["purpose"] = gdf["preceding_purpose"] + " -> " + gdf["following_purpose"]
    gdf["travel_time"] = gdf["arrival_time"] - gdf["departure_time"]
    gdf["origin"] = gdf.geometry.apply(lambda g: Point(g.coords[0]))
    gdf["destination"] = gdf.geometry.apply(lambda g: Point(g.coords[-1]))
    gdf.drop(
        columns=[
            "preceding_activity_index",
            "following_activity_index",
            "is_first",
            "is_last",
            "geometry",
        ],
        inplace=True,
    )
    return gdf


def get_iris_polygons(departements):
    """Returns a gpd.GeoDataFrame with the geometries of the IRIS zones in the selected
    départements.
    """
    print("Reading IRIS zones")
    gdf = gpd.read_file(IRIS_FILE)
    gdf["dep"] = gdf["INSEE_COM"].str[:2]
    gdf = gdf.loc[gdf["dep"].isin(departements)]
    gdf = gdf[["CODE_IRIS", "geometry"]].copy()
    gdf["CODE_IRIS"] = gdf["CODE_IRIS"].astype(int)
    return gdf


def find_iris_origin_destination(trip_gdf, iris_gdf):
    """Returns a gpd.GeoDataFrame of trips with the columns 'iris_origin' and 'iris_destination',
    with the IRIS code for origin and destination of the trip.
    """
    trip_gdf["iris_origin"] = 0
    trip_gdf["iris_destination"] = 0
    # To speed up computation, we discard the already matched points at each
    # iteration.
    origins = gpd.GeoDataFrame(geometry=trip_gdf["origin"])
    origins["x"] = origins.geometry.x
    origins["y"] = origins.geometry.y
    destinations = gpd.GeoDataFrame(geometry=trip_gdf["destination"])
    destinations["x"] = destinations.geometry.x
    destinations["y"] = destinations.geometry.y
    print("Finding trips' IRIS zone")
    n = len(iris_gdf) // 99
    for i, (_, row) in enumerate(iris_gdf.iterrows()):
        if i % n == 0:
            print("{} %".format(i // n))
        (x0, y0, x1, y1) = row["geometry"].bounds
        prepared_geometry = shapely.prepared.prep(row["geometry"])
        # Match origins.
        tmp = origins.loc[
            (origins["x"] >= x0)
            & (origins["x"] <= x1)
            & (origins["y"] >= y0)
            & (origins["y"] <= y1)
        ]
        matches = tmp.loc[map(prepared_geometry.intersects, tmp.geometry)].index
        trip_gdf.loc[matches, "iris_origin"] = row["CODE_IRIS"]
        origins.drop(matches, inplace=True)
        # Match destinations.
        tmp = destinations.loc[
            (destinations["x"] >= x0)
            & (destinations["x"] <= x1)
            & (destinations["y"] >= y0)
            & (destinations["y"] <= y1)
        ]
        matches = tmp.loc[map(prepared_geometry.intersects, tmp.geometry)].index
        trip_gdf.loc[matches, "iris_destination"] = row["CODE_IRIS"]
        destinations.drop(matches, inplace=True)
    return trip_gdf


if __name__ == "__main__":
    trip_gdf = get_trips()
    iris_gdf = get_iris_polygons(IDF_DEP)
    trip_gdf = find_iris_origin_destination(trip_gdf, iris_gdf)
    # Set source and target node id of the trips.
    zone_mapping = get_zone_mapping()
    trip_gdf = (
        trip_gdf.merge(
            zone_mapping.loc[zone_mapping["in"], ["node_id", "CODE_IRIS"]],
            left_on="iris_destination",
            right_on="CODE_IRIS",
        )
        .rename(columns={"node_id": "destination_id"})
        .drop(columns="CODE_IRIS")
    )
    trip_gdf = (
        trip_gdf.merge(
            zone_mapping.loc[~zone_mapping["in"], ["node_id", "CODE_IRIS"]],
            left_on="iris_origin",
            right_on="CODE_IRIS",
        )
        .rename(columns={"node_id": "origin_id"})
        .drop(columns="CODE_IRIS")
    )
    # Merge.
    household_df = get_households()
    person_df = get_persons()
    trip_gdf = trip_gdf.merge(person_df, on="person_id", how="left")
    trip_gdf = trip_gdf.merge(household_df, on="household_id", how="left")
    trip_gdf.drop(columns=["origin", "destination"], inplace=True)
    print("Writing data")
    trip_gdf.to_csv(OUTPUT_FILENAME, index=False)

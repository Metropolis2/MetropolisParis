from shapely.geometry import LineString
import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the file where the trips are stored.
TRIPS_FILE = "./output/trips_filtered.csv"
# Path to the IRIS Shapefile.
IRIS_FILE = "/home/ljavaudin/Projects/MetropolisIDF/data/contours_iris_france/"
# Départements in the studied area, used to filter IRIS zones.
DEPARTEMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]


def get_trips():
    return pd.read_csv(TRIPS_FILE, dtype={"iris_origin": "str", "iris_destination": "str"})


def get_iris_centroids():
    """Returns a gpd.GeoDataFrame with the geometries of the IRIS zones in the selected
    départements.
    """
    print("Reading IRIS zones")
    gdf = gpd.read_file(IRIS_FILE)
    if DEPARTEMENTS:
        gdf = gdf.loc[gdf["INSEE_COM"].str[:2].isin(DEPARTEMENTS)]
    gdf = gdf[["CODE_IRIS", "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.centroid
    gdf.to_crs("epsg:4326", inplace=True)
    gdf.set_index("CODE_IRIS", inplace=True)
    return gdf


def find_destinations(iris_code, trips, iris_centroids):
    centroid = iris_centroids.loc[iris_code].geometry
    dests = trips.loc[trips["iris_origin"] == iris_code, "iris_destination"].value_counts()
    linestrings = [
        LineString([[centroid.x, centroid.y], [row["geometry"].x, row["geometry"].y]])
        for _, row in iris_centroids.loc[dests.index].iterrows()
    ]
    gdf = gpd.GeoDataFrame(
        data={"destination": dests.index, "count": dests.values},
        geometry=linestrings,
        crs="epsg:4326",
    )
    return gdf


def find_origins(iris_code, trips, iris_centroids):
    centroid = iris_centroids.loc[iris_code].geometry
    origins = trips.loc[trips["iris_destination"] == iris_code, "iris_origin"].value_counts()
    linestrings = [
        LineString([[row["geometry"].x, row["geometry"].y], [centroid.x, centroid.y]])
        for _, row in iris_centroids.loc[origins.index].iterrows()
    ]
    gdf = gpd.GeoDataFrame(
        data={"origin": origins.index, "count": origins.values},
        geometry=linestrings,
        crs="epsg:4326",
    )
    return gdf

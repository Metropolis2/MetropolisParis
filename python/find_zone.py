from shapely.geometry import Point
import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the file where the mapping between IRIS zone ids and node ids is stored.
ZONE_ID_FILE = "./output/zone_id_map.csv"
# Path to the IRIS Shapefile.
IRIS_FILE = "/home/ljavaudin/Projects/MetropolisIDF/data/contours_iris_france/"
# Départements in the studied area, used to filter IRIS zones.
DEPARTEMENTS = ["75"]


def get_zone_mapping():
    df = pd.read_csv(ZONE_ID_FILE, dtype={'CODE_IRIS': 'str'})
    return df.set_index(['CODE_IRIS', 'in'])


def get_iris_polygons():
    """Returns a gpd.GeoDataFrame with the geometries of the IRIS zones in the selected
    départements.
    """
    print("Reading IRIS zones")
    gdf = gpd.read_file(IRIS_FILE)
    if DEPARTEMENTS:
        gdf = gdf.loc[gdf["INSEE_COM"].str[:2].isin(DEPARTEMENTS)]
    gdf = gdf[["CODE_IRIS", "geometry"]].copy()
    gdf.to_crs('epsg:4326', inplace=True)
    return gdf


def find_zone(x, y, iris_gdf, zone_mapping, origin=True):
    """Returns the IRIS and Metropolis zone from (x, y) coordinates."""
    point = Point(x, y)
    valid_zones = np.flatnonzero(iris_gdf.contains(point))
    if len(valid_zones) == 0:
        print('The point is not in an IRIS zone')
    elif len(valid_zones) > 1:
        print('The point is in multiple IRIS zones')
    else:
        iris_zone = iris_gdf.loc[valid_zones[0], 'CODE_IRIS']
        print('IRIS zone: {}'.format(iris_zone))
        metro_zone = zone_mapping.loc[(iris_zone, origin), 'node_id']
        print('Metropolis zone: {}'.format(metro_zone))
        return iris_zone, metro_zone

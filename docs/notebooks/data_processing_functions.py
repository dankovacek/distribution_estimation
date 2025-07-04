import os
from time import time
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy.stats import entropy, wasserstein_distance, linregress
# from sklearn.metrics import (
#     root_mean_squared_error,
#     mean_absolute_error,
#     f1_score, d2_log_loss_score, confusion_matrix, fbeta_score
# )
import xarray as xr
from sklearn.model_selection import StratifiedKFold
from shapely.geometry import Point

import xgboost as xgb
import multiprocessing as mp
import jax
import jax.numpy as jnp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def count_complete_years(df, date_column, value_column):
    # Convert to datetime only if necessary
    if not np.issubdtype(df[date_column].dtype, np.datetime64):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

    # Filter out missing values first
    valid_data = df[df[value_column].notna()]
    
    # Extract year and month
    valid_data['year'] = valid_data[date_column].dt.year
    valid_data['month'] = valid_data[date_column].dt.month
    valid_data['day'] = valid_data[date_column].dt.day
    
    # Count total and missing days per year-month group
    month_counts = valid_data.groupby(['year', 'month'])['day'].nunique()
    
    # Identify complete months (assuming a month has up to 31 days)
    complete_months = (month_counts >= (0.9 * month_counts.groupby(level=0).transform('max')))

    # Count years where all 12 months are complete
    complete_years = complete_months.groupby(level=0).sum()
    return (complete_years == 12).sum()


def load_and_filter_streamflow_timeseries(station_ids, hs_df, HYSETS_DIR):

    # load the updated HYSETS data
    updated_filename = 'HYSETS_2023_update_QC_stations.nc'
    ds = xr.open_dataset(HYSETS_DIR / updated_filename)

    # Get valid IDs as a NumPy array
    hs_df = hs_df[hs_df['Official_ID'].isin(station_ids)]
    selected_ids = hs_df['Watershed_ID'].values

    # Get boolean index where watershedID in selected_set
    # safely access watershedID as a variable first
    ws_ids = ds['watershedID'].data  # or .values if you prefer
    mask = np.isin(ws_ids, selected_ids)

    # Apply mask to data
    ds = ds.sel(watershed=mask)
    # Step 1: Promote 'watershedID' to a coordinate on the 'watershed' dimension
    ds = ds.assign_coords(watershedID=("watershed", ds["watershedID"].data))

    # Step 2: Set 'watershedID' as the index for the 'watershed' dimension
    return ds.set_index(watershed="watershedID")


def load_study_region_stations(study_region_station_fpath):
    # import the BCUB (study) region boundary
    # the file can be found at https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/JNKZVT
    # it may be labeled BCUB_regions_4326.geojson
    # region_gdf = gpd.read_file('data/BCUB_regions_merged_R0.geojson')
    # region_gdf = region_gdf.to_crs(3005)
    # # simplify the geometries (100m threshold) and add a small buffer (250m) 
    # # to capture HYSETS station points recorded with low accuracy near boundaries
    # region_gdf.geometry = region_gdf.simplify(100).buffer(500)
    # region_gdf = region_gdf.to_crs(4326)
    # get the stations contained in the study region
    # centroids = hysets_df.apply(lambda x: Point(x['Centroid_Lon_deg_E'], x['Centroid_Lat_deg_N']), axis=1)
    # hysets_gdf = gpd.GeoDataFrame(hysets_df, geometry=centroids, crs='EPSG:4326')
    # hysets_gdf.head(4)
    # assert hysets_gdf.crs == region_gdf.crs

    # bcub_gdf = gpd.sjoin(hysets_gdf, region_gdf, how='inner', predicate='intersects')
    # print(len(bcub_gdf), len(set(bcub_gdf['Official_ID'])))

    # # Because of the buffer (to capture stations along the coast), 
    # # there's a duplicated 08GA065 that should be in 08G
    # bcub_gdf = bcub_gdf.drop_duplicates(subset=['Official_ID'])
    # two stations in the far north lie just outside the study region but may be
    # valuable to include given the low density of monitoring in the region.
    # to_include = ['10ED002', '09AG003']
    # added_stns = hysets_df[hysets_df['Official_ID'].isin(to_include)].copy()
    # added_centroids = added_stns.apply(lambda x: Point(x['Centroid_Lon_deg_E'], x['Centroid_Lat_deg_N']), axis=1)
    # added_gdf = gpd.GeoDataFrame(added_stns, geometry=added_centroids, crs='EPSG:4326')
    # bcub_gdf = gpd.GeoDataFrame(pd.concat([bcub_gdf, added_gdf]), crs='4326')
    # bcub_gdf.loc[bcub_gdf['Official_ID'] == '10ED002', 'region_code'] = '10E'
    # bcub_gdf.loc[bcub_gdf['Official_ID'] == '09AG003', 'region_code'] = 'YKR'
    # bcub_gdf.to_file(study_region_stations_fpath)
    assert os.path.exists(study_region_station_fpath), f"File not found: {study_region_station_fpath}"
    bcub_gdf = gpd.read_file(study_region_station_fpath)    # get the number of unique stations in the dataset
    return bcub_gdf



def format_fig_fonts(fig, font_size=20, font='Bitstream Charter', legend=True):
    fig.xaxis.axis_label_text_font_size = f'{font_size}pt'
    fig.yaxis.axis_label_text_font_size = f'{font_size}pt'
    fig.xaxis.major_label_text_font_size = f'{font_size-2}pt'
    fig.yaxis.major_label_text_font_size = f'{font_size-2}pt'
    fig.yaxis.axis_label_text_font = font
    fig.xaxis.axis_label_text_font = font
    fig.xaxis.major_label_text_font = font
    fig.yaxis.major_label_text_font = font
    if fig.legend == True:
        fig.legend.label_text_font_size = f'{font_size-2}pt'
        fig.legend.label_text_font = font
    return fig
    
# Add custom CSS for styling (optional)
table_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400&display=swap');

.styled-table {
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 0.9em;
    font-family: Roboto, sans-serif;
    width: 100%;
    text-align: left;
}
.styled-table th {
    background-color: white;
    color: black;
    padding: 12px 15px;
}
.styled-table td {
    padding: 12px 15px;
}
.styled-table tr {
    border-bottom: 1px solid #dddddd;
}
.styled-table tr:nth-of-type(even) {
    background-color: #f3f3f3;
}
.styled-table tr:last-of-type {
    border-bottom: 2px solid #009879;
}
</style>
"""

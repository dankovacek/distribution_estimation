import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from numba import jit
import jax.numpy as jnp

from pathlib import Path

HYSETS_DIR = Path('/home/danbot/code/common_data/HYSETS')


@jit
def compute_overlap_matrix(mask_jax):
    return jnp.matmul(mask_jax, mask_jax.T)


class FDCEstimationContext:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._load_catchment_data()
        self._load_and_filter_hysets_data()
        self.LN_param_dict = self._get_ln_params()
        self._set_tree_idx_mappers()
        self._build_network_trees()
        self._set_attribute_indexers()
        self.overlap_dict = self._compute_concurrent_overlap_dict()
        
    
    def _load_and_filter_hysets_data(self):
        hs_df = pd.read_csv('data/HYSETS_watershed_properties.txt', sep=';')
        if self.LSTM_concurrent_network == True:
            self.global_start_date = pd.to_datetime('1980-01-01') # Daymet starts 1980-01-01
        else:            
            self.global_start_date = pd.to_datetime('1950-01-01') # Hysets streamflow starts 1950-01-01
        hs_df = hs_df[hs_df['Official_ID'].isin(self.official_ids)]
        self.hs_df = hs_df
        self.official_ids = hs_df['Official_ID'].values
        print(f'Use only stations with minimum concurrency with Daymet / LSTM results: {self.LSTM_concurrent_network} (n={len(self.official_ids)})')

        # load the updated HYSETS data
        updated_filename = 'HYSETS_2023_update_QC_stations.nc'
        ds = xr.open_dataset(HYSETS_DIR / updated_filename)

        # Get valid IDs as a NumPy array
        selected_ids = hs_df['Watershed_ID'].values

        # Get boolean index where watershedID in selected_set
        # safely access watershedID as a variable first
        ws_ids = ds['watershedID'].data  # or .values if you prefer
        mask = np.isin(ws_ids, selected_ids)

        # Apply mask to data
        ds = ds.sel(watershed=mask)
        # Step 1: Promote 'watershedID' to a coordinate on the 'watershed' dimension
        ds = ds.assign_coords(watershedID=("watershed", ds["watershedID"].data))

        # filter the timeseries by the global start date
        ds = ds.sel(time=slice(self.global_start_date, None))

        # Step 2: Set 'watershedID' as the index for the 'watershed' dimension
        self.ds = ds.set_index(watershed="watershedID")
    
    
    def _load_catchment_data(self):
        gdf = gpd.read_file(self.attr_gdf_fpath)
        gdf.columns = [c.lower() for c in gdf.columns]

        # LSTM ensemble predictions
        # lstm_result_folder = '/home/danbot/code/neuralhydrology/data/ensemble_results'
        # lstm_result_files = os.listdir(lstm_result_folder)
        # lstm_result_stns = [e.split('_')[0] for e in lstm_result_files]

        # filter for the common stations between BCUB region and LSTM-compatible (i.e. 1980-)
        if self.LSTM_concurrent_network == True:
            self.official_ids = self.daymet_concurrent_stations
        else:
            self.official_ids = self.all_official_ids 

        gdf = gdf[gdf['official_id'].isin(self.official_ids)]
        # import the license water extraction points
        dam_gdf = gpd.read_file('data/Dam_Points_20240103.gpkg')
        assert gdf.crs == dam_gdf.crs, "CRS of catchment and dam dataframes do not match"
        joined = gpd.sjoin(gdf, dam_gdf, how="inner", predicate="contains")
        joined = joined[joined['official_id'].isin(self.official_ids)]
        # Create a new boolean column 'contains_dam' in catchment_gdf.
        # If a polygon's index appears in the joined result, it means it contains at least one point.
        regulated = joined['official_id'].values
        N = len(gdf)
        print(f'{len(regulated)}/{N} catchments contain withdrawal licenses')
                
        # create dict structures for easier access of attributes and geometries
        self.da_dict = gdf[['official_id', 'drainage_area_km2']].set_index('official_id').to_dict()['drainage_area_km2']
        self.polygon_dict = gdf[['official_id', 'geometry']].set_index('official_id').to_dict()['geometry']
        
        centroids = gdf.apply(lambda x: self.polygon_dict[x['official_id']].centroid, axis=1)
        attr_gdf = gpd.GeoDataFrame(gdf, geometry=centroids, crs=gdf.crs)
        print(len(attr_gdf))
        attr_gdf["contains_dam"] = attr_gdf['official_id'].apply(lambda x: x in regulated)
        attr_gdf.reset_index(inplace=True, drop=True)
        attr_gdf['tmean'] = (attr_gdf['tmin'] + attr_gdf['tmax']) / 2.0
        attr_gdf['log_drainage_area_km2'] = np.log(attr_gdf['drainage_area_km2'])
        self.attr_gdf = attr_gdf


    def _build_network_trees(self, attribute_cols=['log_drainage_area_km2', 'elevation_m', 'prcp', 'tmean', 'swe', 'srad',
                            'centroid_lon_deg_e', 'centroid_lat_deg_n', 'land_use_forest_frac_2010', 'land_use_snow_ice_frac_2010',
                            #  , 'land_use_wetland_frac_2010', 'land_use_water_frac_2010', 
                            ]):
        self.coords = np.array([[geom.x, geom.y] for geom in self.attr_gdf.geometry.values])
        self.spatial_tree = cKDTree(self.coords)

        # Extract values (excluding 'official_id' since it's categorical)
        attr_values = self.attr_gdf[attribute_cols].to_numpy()
        scaler = StandardScaler()
        self.normalized_attr_values = scaler.fit_transform(attr_values)
        # Construct the KDTree for the attribute space
        self.attribute_tree = cKDTree(self.normalized_attr_values)


    def _set_tree_idx_mappers(self):
        """Set the index mappers for the official_id to index and vice versa.
        This is for the network TREES"""
        self.id_to_idx = {id: i for i, id in enumerate(self.attr_gdf['official_id'].values)}
        self.idx_to_id = {i: id for i, id in enumerate(self.attr_gdf['official_id'].values)}
    

    def _set_attribute_indexers(self):
        
        # Create mapping from official_id to index
        # self.id_to_index = {oid: i for i, oid in enumerate(self.attr_gdf["official_id"])}
        # self.index_to_id = {i: oid for oid, i in self.id_to_index.items()}  # Reverse mapping
        # create a dictionary where the keys are Watershed_ID and the values are Official_ID
        self.watershed_id_dict = {row['Watershed_ID']: row['Official_ID'] for _, row in self.hs_df.iterrows()}
        # and the inverse
        self.official_id_dict = {row['Official_ID']: row['Watershed_ID'] for _, row in self.hs_df.iterrows()}
        # also for drainage areas
        self.da_dict = {row['Official_ID']: row['Drainage_Area_km2'] for _, row in self.hs_df.iterrows()}
        

    def _get_ln_params(self):
        """Retrieve log-normal parameters for a station."""
        parameter_prediction_results_folder = os.path.join('data', 'parameter_prediction_results', )
        predicted_params_fpath   = os.path.join(parameter_prediction_results_folder, 'mean_parameter_predictions.csv')
        rdf = pd.read_csv(predicted_params_fpath, index_col=['official_id'], dtype={'official_id': str})
        rdf.columns = ['_'.join(c.split('_')[:-1]) for c in rdf.columns]
        return rdf.to_dict(orient='index')
    

    def _generate_12_month_windows(self, index):
        months = pd.date_range(index.min(), index.max(), freq='MS')
        windows = [(start, start + pd.DateOffset(months=12) - pd.Timedelta(days=1)) for start in months]
        return [w for w in windows if w[1] <= index.max()]
    
    
    def _is_window_valid(self, ts, start, end):
        window = ts.loc[start:end]
        if window.empty:
            return False
        grouped = window.groupby(window.index.month).size()
        if set(grouped.index) != set(range(1, 13)):
            return False
        if grouped.min() < 10:
            return False
        return True
    
    
    def _compute_station_valid_windows(self, ts, windows):
        return [self._is_window_valid(ts, start, end) for (start, end) in windows]
    
    
    def _count_valid_shared_windows(self, valid_i, valid_j):
        return sum(np.logical_and(valid_i, valid_j))
    
       
    def _compute_concurrent_overlap_dict(self, variable='discharge'):
        """
        Compute the concurrent overlap of monitored watersheds in the dataset.
        Threshold years represent the minimum number of days of overlap 
        (ignoring continuity) for a watershed to be considered concurrent.
        """
        overlap_dict_file = os.path.join('data', 'record_overlap_dict.json')
        if os.path.exists(overlap_dict_file):
            with open(overlap_dict_file, 'r') as f:
                overlap_dict = json.load(f)
            print(f'    ...overlap dict loaded from {overlap_dict_file}')
            return overlap_dict
        
        watershed_ids = self.ds['watershed'].values
        data = self.ds[variable].load().values  # (N, T)
        dates = pd.to_datetime(self.ds['time'].values) # shape (T, )
        N, T = data.shape

        # Extract year/month info 
        years = np.array([d.year for d in dates])
        months = np.array([d.month for d in dates])
        unique_years = np.unique(years)
        year_to_index = {year: i for i, year in enumerate(unique_years)}
        Y = len(unique_years)

        # Count valid obs per (year, site, month) 
        monthly_counts = np.zeros((Y, N, 12), dtype=np.uint16)
        valid_mask = ~np.isnan(data)  # shape (N, T)

        for t in range(T):
            y_idx = year_to_index[years[t]]
            m_idx = months[t] - 1  # month index in [0, 11]
            monthly_counts[y_idx, :, m_idx] += valid_mask[:, t]

        # Create monthly validity mask
        monthly_valid = monthly_counts >= self.minimum_days_per_month  # shape: (Y, N, 12)
        valid_years = np.all(monthly_valid, axis=-1)  # shape: (Y, N)
        complete_years = np.sum(valid_years, axis=0)  # shape: (N,)

        # Compute joint validity per pair
        joint_valid = np.logical_and(
            monthly_valid[:, :, None, :],  # shape: (Y, i, 1, M)
            monthly_valid[:, None, :, :]   # shape: (Y, 1, j, M)
        )  # shape: (Y, i, j, M)

        joint_year_valid = np.all(joint_valid, axis=-1)               # shape: (Y, i, j)
        concurrent_years_matrix = np.sum(joint_year_valid, axis=0)    # shape: (i, j)
        
        # loop over minimum concurrent proportions
        overlap_dict = defaultdict(dict)
        for prop in self.min_target_overlap_proportions:
            # Mask where concurrent years ≥ required proportion of complete years for i
            thresholds = (complete_years * prop / 100.0).astype(int)  # shape: (N,)
            valid_mask = concurrent_years_matrix >= thresholds[:, None]  # broadcast shape: (N, N)
            np.fill_diagonal(valid_mask, False)

            # Build overlap_dict
            odict = {
                int(watershed_ids[i]): list(map(int, watershed_ids[valid_mask[i]]))
                for i in range(N)
            }
            overlap_dict[str(prop)] = odict

        with open(overlap_dict_file, 'w') as f:
            json.dump(overlap_dict, f)
        print(f'    ...saved overlap dict to {overlap_dict_file}')
        return overlap_dict
        
        
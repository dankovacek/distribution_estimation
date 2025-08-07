# import data_processing_functions as dpf
# from concurrent.futures import ThreadPoolExecutor
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from .kde_estimator import KDEEstimator

class LSTMFDCEstimator:
    def __init__(self, ctx, target_stn, data, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.target_stn = target_stn
        self.data = data
        # self.data = data
        self.LSTM_forcings_folder = self.ctx.LSTM_forcings_folder
        self.LSTM_ensemble_result_folder = self.ctx.LSTM_ensemble_result_folder
        self.df = self._load_ensemble_result()
        self.df = self._filter_for_complete_years()
        self.sim_cols = sorted([c for c in self.df.columns if c.startswith('streamflow_sim_')])
        self.kde = KDEEstimator(self.data.baseline_log_grid, self.data.log_dx)


    def _load_ensemble_result(self):
        fpath = os.path.join(self.LSTM_ensemble_result_folder, f'{self.target_stn}_ensemble.csv')
        df = pd.read_csv(fpath)
        # rename 'Unnamed: 0' to 'time' and set to index
        df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    

    def _filter_for_complete_years(self):
        # Convert to datetime only if necessary
        if self.df.empty:
            return pd.DataFrame()
        date_column = 'time'
        self.df.reset_index(inplace=True)
        if not np.issubdtype(self.df[date_column].dtype, np.datetime64):
            self.df = self.df.copy()
            self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Filter out missing values first
        valid_data = self.df.copy().dropna()

        # Extract year and month
        valid_data['year'] = valid_data[date_column].dt.year
        valid_data['month'] = valid_data[date_column].dt.month
        valid_data['day'] = valid_data[date_column].dt.day
        
        # Count total and missing days per year-month group
        month_counts = valid_data.groupby(['year', 'month'])['day'].nunique()
        
        # Identify complete months (at least 20 observations)
        complete_months = (month_counts >= 20)

        # count how many complete months per year
        complete_month_counts = complete_months.groupby(level=0).sum()
        
        complete_years = complete_month_counts[complete_month_counts == 12]
        self.complete_years = list(complete_years.index.values)

        valid_data = valid_data[valid_data['year'].isin(complete_years.index)].copy()
        # drop the year column
        return valid_data.drop(columns=['year', 'month', 'day'])
    
    
    def _load_LSTM_forcing_file(self):
        # retrieve LSTM forcing data
        # read the forcing data from the LSTM forcing file
        # and return a dataframe with the same index as the LSTM results
        ldf = pd.read_csv(os.path.join(self.met_forcings_folder, f'{self.target_stn}_forcing.csv'))
        ldf.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        ldf['time'] = pd.to_datetime(ldf['time'])
        ldf.set_index('time', inplace=True)
        ldf = ldf.loc[self.stn_df.index]
        # convert to unit area runoff (L/s/km2)
        ldf['uar'] = 1000 * ldf['discharge'] / self.target_da
        return ldf

    
    def _plot_pmfs(self, pmf_time, pmf_freq, line_dash='solid'):
        # plot using bokeh
        f = figure(title=self.target_stn, width=600, height=400)
        f.line(self.data.baseline_log_grid, pmf_time, line_width=2, color='blue', legend_label='Time Ensemble', line_dash=line_dash)
        # f.line(self.data.baseline_log_grid, pmf1, line_width=2, color='red', legend_label='T_MeanLinEns PMF', line_dash=line_dash)
        f.line(self.data.baseline_log_grid, pmf_freq, line_width=2, color='purple', legend_label='Frequency Ensemble', line_dash=line_dash)
        f.line(self.data.baseline_log_grid, self.ctx.baseline_pmf, line_width=2, color='green', legend_label='Observed', line_dash=line_dash)
        f.xaxis.axis_label = 'Log UAR (L/s/km2)'
        f.yaxis.axis_label = 'PMF'
        f.legend.location = 'top_left'
        f.legend.background_fill_alpha = 0.25
        f.legend.click_policy = 'hide'
        f = dpf.format_fig_fonts(f, font_size=14)
        show(f)


    def _compute_time_ensemble_pmf(self):
        data = self.df[self.sim_cols].copy()
        temporal_ensemble_log = data.mean(axis=1) # this is still in log space
        self.temporal_ensemble = np.exp(temporal_ensemble_log.values)
        pmf, _ = self.kde.compute(self.temporal_ensemble, self.data.target_da)
        _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_laplace_prior(pmf)
        return (pmf, prior_adjusted_pmf)


    def _compute_frequency_ensemble_pmf(self):
        data = self.df[self.sim_cols].copy()
        data.dropna(inplace=True)
        # compute the frequency ensemble PMF
        # initialize a len(data) x n_sim_cols array
        pmfs = np.column_stack([
            self.kde.compute(np.exp(data[c].values), self.data.target_da)[0]
            for c in self.sim_cols
        ])
        # average the pmfs over the ensemble 
        pmf = pmfs.mean(axis=1)
        assert len(pmf) == len(self.data.baseline_log_grid), f'len(pmfs) = {len(pmfs)} != len(baseline_log_grid) = {len(self.data.baseline_log_grid)}' 
        _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_laplace_prior(pmf)
        return (pmf, prior_adjusted_pmf)


    def _compute_ensemble_distribution_estimate(self, ensemble_type):
        if ensemble_type == 'time':
            pmf, prior_adjusted_pmf = self._compute_time_ensemble_pmf()
        elif ensemble_type == 'frequency':
            pmf, prior_adjusted_pmf = self._compute_frequency_ensemble_pmf()
        else:
            raise ValueError(f'Unknown ensemble type: {ensemble_type}')
        
        # compute the divergence measures
        result = {}
        result['pmf'] = pmf.tolist()
        result['prior_adjusted_pmf'] = prior_adjusted_pmf.tolist()
        result['eval'] = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, self.data.baseline_pmf)
        result['bias'] = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, pmf)
        return result


    def run_estimators(self):
        # met_forcing = self._load_LSTM_forcing_file()  # Load LSTM forcing data
        results = {}
        for ensemble_type in ['time', 'frequency']:
            # print(f'     Processing {ensemble_type} ensemble for {self.target_stn}')
            result = self._compute_ensemble_distribution_estimate(ensemble_type)
            results[ensemble_type] = result
        return results
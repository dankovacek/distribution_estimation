# import data_processing_functions as dpf
# from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import pandas as pd

class LSTMFDCEstimator:
    def __init__(self, ctx, data, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.data = data
        self.target_stn = self.data.target_stn
        # self.data = data
        self.LSTM_forcings_folder = self.ctx.LSTM_forcings_folder
        self.LSTM_ensemble_result_folder = self.ctx.LSTM_ensemble_result_folder
        self.df = self._load_ensemble_result()
        self.df = self._filter_for_complete_years()
        self.sim_cols = sorted([c for c in self.df.columns if c.startswith('streamflow_sim_')])


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

    
    def _compute_time_ensemble_pmf(self):
        data = self.df[self.sim_cols].copy()
        temporal_ensemble_log = data.mean(axis=1) # this is still in log space
        # self.temporal_ensemble = np.exp(temporal_ensemble_log.values)
        # pmf, _ = self.data.kde_estimator.compute(self.temporal_ensemble, self.data.target_da)
        pmf, _ = np.histogram(temporal_ensemble_log, bins=self.data.log_edges, density=True)
        # normalize the pmf sum to 1
        pmf = pmf / pmf.sum()
        # _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_laplace_prior(pmf)
        _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf)
        return (pmf, prior_adjusted_pmf)
    

    def compute_ensemble_pmf_by_bincount(self, df, cols, edges):
        """Faster computation of ensemble PMF by bin counting."""
        X = df[cols].to_numpy(copy=False); edges = np.asarray(edges); n = edges.size - 1
        b = np.searchsorted(edges, X, 'right') - 1
        v = (b >= 0) & (b < n) & np.isfinite(X)
        flat = b[v] + n * np.nonzero(v)[1]
        C = np.bincount(flat, minlength=n * X.shape[1]).reshape(X.shape[1], n).T
        pmf = C / np.clip(C.sum(0, keepdims=True), 1, None)
        m = pmf.mean(1)
        return m / m.sum() if m.sum() else m



    def _compute_frequency_ensemble_pmf(self):
        df = self.df[self.sim_cols].copy()
        df.dropna(inplace=True)
        # compute the frequency ensemble PMF
        # initialize a len(data) x n_sim_cols array
        # pmfs = np.column_stack([
        #     self.data.kde_estimator.compute(np.exp(data[c].values), self.data.target_da)[0]
        #     for c in self.sim_cols
        # ])
        # compute the pmfs by bin counting over the 
        # average the pmfs over the ensemble 
        # pmfs = np.column_stack([
        #     np.histogram(df[c].values, bins=self.data.log_edges, density=True)[0]
        #     for c in self.sim_cols
        # ])
        # pmf = pmfs.mean(axis=1)
        # pmf /= pmf.sum()
        pmf = self.compute_ensemble_pmf_by_bincount(df, self.sim_cols, self.data.log_edges)

        assert len(pmf) == len(self.data.log_x), f'len(pmf) = {len(pmf)} != len(log_x) = {len(self.data.log_x)}' 
        # _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_laplace_prior(pmf)
        _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf)
        
        return (pmf, prior_adjusted_pmf)


    def _compute_ensemble_distribution_estimate(self, ensemble_type):
        if ensemble_type == 'time':
            pmf, prior_adjusted_pmf = self._compute_time_ensemble_pmf()
        elif ensemble_type == 'frequency':
            pmf, prior_adjusted_pmf = self._compute_frequency_ensemble_pmf()
        else:
            raise ValueError(f'Unknown ensemble type: {ensemble_type}')
        
        result = {}
        result['pmf'] = pmf.tolist()
        result['prior_adjusted_pmf'] = prior_adjusted_pmf.tolist()
        result['eval'] = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, self.data.baseline_obs_pmf)
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
    

    # def _plot_pmfs(self, pmf_time, pmf_freq, line_dash='solid'):
        # plot using bokeh
        # f = figure(title=self.target_stn, width=600, height=400)
        # f.line(self.data.log_x, pmf_time, line_width=2, color='blue', legend_label='Time Ensemble', line_dash=line_dash)
        # # f.line(self.data.log_x, pmf1, line_width=2, color='red', legend_label='T_MeanLinEns PMF', line_dash=line_dash)
        # f.line(self.data.log_x, pmf_freq, line_width=2, color='purple', legend_label='Frequency Ensemble', line_dash=line_dash)
        # f.line(self.data.log_x, self.ctx.baseline_obs_pmf, line_width=2, color='green', legend_label='Observed', line_dash=line_dash)
        # f.xaxis.axis_label = 'Log UAR (L/s/km2)'
        # f.yaxis.axis_label = 'PMF'
        # f.legend.location = 'top_left'
        # f.legend.background_fill_alpha = 0.25
        # f.legend.click_policy = 'hide'
        # f = dpf.format_fig_fonts(f, font_size=14)
        # show(f)


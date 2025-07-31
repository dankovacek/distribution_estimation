from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import laplace
import json

# from utils.evaluation_metrics import EvaluationMetrics

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import wasserstein_distance

@dataclass
class StationData:
    def __init__(self, context, stn):
        self.ctx = context
        self.target_stn = stn
        self.attr_gdf = context.attr_gdf
        self.predicted_param_dict = context.LN_param_dict[stn]
        self.n_grid_points = context.n_grid_points
        # self.catchments = catchments
        self.min_flow = context.min_flow # don't allow flows below this value
        self.divergence_measures = context.divergence_measures
        self.met_forcings_folder = context.LSTM_forcings_folder
        self.LSTM_ensemble_result_folder = context.LSTM_ensemble_result_folder

        self.target_da = float(self.attr_gdf[self.attr_gdf['official_id'] == stn]['drainage_area_km2'].values[0])
        self._initialize_target_streamflow_data()
        # self._set_grid()
        # self.eval_metrics = EvaluationMetrics(self.baseline_log_grid, self.log_dx)
        # self._set_divergence_measure_functions()
        self._load_baseline_distribution()
        self._load_complete_year_dict()
        self.location = self.ctx.laplace_param_dict['median'][stn]
        self.scale = self.ctx.laplace_param_dict['mad'][stn]
        self.prior_strength = self.ctx.prior_strength



    def _load_complete_year_dict(self):
        """
        Load the complete year dictionary for the target station.
        This is used to ensure that the time series data is complete for the target station.
        """
        with open('data/complete_years.json', 'r') as f:
            self.complete_year_dict = json.load(f)

    
    def _load_baseline_distribution(self):
        """
        Set the baseline distribution for the target station.
        This is used to compare the simulated distributions against the observed distribution.
        """
        # load the baseline PMF for the target station
        self.baseline_pmf = self.ctx.baseline_pmf_df[self.target_stn].values
        assert np.isclose(np.sum(self.baseline_pmf), 1), f'Baseline PMF for {self.target_stn} does not sum to 1: {np.sum(self.baseline_pmf)}'
        self.baseline_pdf = self.ctx.baseline_pdf_df[self.target_stn].values
        eval_grid = self.ctx.baseline_pdf_df.index.values
        # compute the PDF from the PMF given the linear grid (index of the pdf_df)
        self.baseline_pdf = self.baseline_pmf / eval_grid
        pdf_area = np.trapezoid(self.baseline_pdf, x=eval_grid)
        self.baseline_pdf /= pdf_area  # normalize the PDF to sum to 1
        assert np.isclose(np.trapezoid(self.baseline_pdf, x=eval_grid), 1), f'Baseline PDF for {self.target_stn} does not sum to 1: {pdf_area}'
        self.baseline_lin_grid = self.ctx.baseline_pmf_df.index.values
        self.baseline_log_grid = np.log(self.baseline_lin_grid)
        self.log_dx = np.gradient(self.baseline_log_grid)


    def retrieve_timeseries_discharge(self, stn):
        watershed_id = self.ctx.official_id_dict[stn]
        df = self.ctx.ds['discharge'].sel(watershed=str(watershed_id)).to_dataframe(name='discharge').reset_index()
        df = df.set_index('time')[['discharge']]
        # regardless of whether the INPUT data is clipped to 
        # match the LSTM input data, the target data should be clipped
        # such that the target distribution is held constant
        df['zero_flow_flag'] = df['discharge'] == 0
        df = df[df.index >= pd.to_datetime(self.ctx.global_start_date)]
        df.dropna(inplace=True)
        # clip minimum flow to 1e-4
        df['discharge'] = np.clip(df['discharge'], self.ctx.min_flow, None)
        df.rename(columns={'discharge': stn}, inplace=True)
        df[f'{stn}_uar'] = 1000 * df[stn] / self.ctx.da_dict[stn]      
        return df
        

    def _initialize_target_streamflow_data(self):
        # self.stn_df = dpf.get_timeseries_data(self.target_stn)
        self.stn_df = self.retrieve_timeseries_discharge(self.target_stn)
        self.uar_label = f'{self.target_stn}_uar'
        self.n_observations = len(self.stn_df[self.uar_label].dropna())  
    

    # def _set_grid(self):        
        # self.baseline_log_grid = np.linspace(np.log(adjusted_min_uar), np.log(max_uar), self.n_grid_points)
        # min_q, max_q = 1e1, 1e6
        # log_min, log_max = np.log(min_q), np.log(max_q)
        # self.baseline_log_grid = np.linspace(log_min, log_max, self.n_grid_points)
        # self.baseline_lin_grid = np.exp(self.baseline_log_grid)
        # self.log_dx = np.gradient(self.baseline_log_grid)
        # max_step_size = self.baseline_lin_grid[-1] - self.baseline_lin_grid[-2]
        # print(f'    max step size = {max_step_size:.1f} L/s/km^2 for n={self.n_grid_points:.1e} grid points')        
            

    def _adjust_Q_pdf_with_prior(self, Q, label):
        """
        Adjusts the simulated PDF Q(x), originally defined on x_sim, by incorporating a Dirichlet prior,
        to produce an adjusted (posterior) PDF Q_mod on x_target.

        """
        # Ensure the target grid lies within the global range.
        # Compute the global log-width.
                        
        # Convert Q_interp (a PDF) into effective "counts" using the number of observations.
        # counts_Q = N_obs * Q
        
        # Convert the prior density into pseudo-counts.
        # years_equiv = n_series * (N_obs / 365.25)
        # counts_prior = (self.pdf_prior / years_equiv) * dx
        # prior_pdf_interp = jnp.interp(self.baseline_log_grid, self.ba, prior_pdf, left=0, right=0)
        prior_pseudo_counts = self.knn_simulation_data[label]['prior']
        n_observations = self.knn_simulation_data[label]['n_obs']
        
        # Combine the counts from Q and the prior.
        Q_mod = Q * n_observations + prior_pseudo_counts
        # Renormalize to obtain the adjusted PDF (discrete PMF) on x_target.
        Q_mod /= np.sum(Q_mod)
        assert np.all(np.isfinite(Q_mod)), 'Q_mod is messed up'
        if not np.min(Q_mod) > 0:
            print('Q_mod min:', np.min(Q_mod))
            print('Q_mod sum:', np.sum(Q_mod))
            print('Q_mod:', Q_mod)
            print('Q:', Q)
            print('prior_pseudo_counts:', prior_pseudo_counts)
            print('n_observations:', n_observations)
            Q_mod += 1e-18
            Q_mod /= np.sum(Q_mod)
            # raise ValueError(f'Q_mod min < 0: {np.min(Q_mod)}')
        assert np.min(Q_mod) > 0, f'qmod_i < 0 ({np.min(Q_mod)})'
        assert np.isclose(np.sum(Q_mod), 1), f"Q_mod doesn't sum to 1: {np.sum(Q_mod):.5f}"

        q_mask = (Q > 0)
        prior_bias = jnp.sum(jnp.where(q_mask, Q * jnp.log2(Q / Q_mod), 0))
        if prior_bias < -0.0001:
            prior_pdf = self.knn_simulation_data[label]['prior']
            print('prior pmf sum =', np.sum(prior_pdf))
            print('Q_sum = ', np.sum(Q))
            print('Q_mod sum = ', np.sum(Q_mod))
            msg = f'    Prior bias {prior_bias:.3f} bits/sample bias'
            raise ValueError(msg)
        
        return Q_mod
    
    
    def _compute_prior_from_laplace_fit(self, location, scale):
        """
        Fit a Laplace distribution to the simulation and define a 
        pdf across a pre-determined "global" range to avoid data
        leakage.  "Normalize" by setting the total prior mass to
        integrate to a factor related to the number of observations.

        The location of the Laplace distribution is the median, 
        and the scale is the mean absolute deviation (MAD).
        By Jensen's Inequality, the MAD is less than or equal to the standard deviation.
        Here we just use the predicted log-mean and log-standard deviation
        as an approximation of the Laplace distribution parameters.
        """
        prior_pdf = laplace.pdf(self.baseline_log_grid, loc=location, scale=scale)
        prior_check = np.trapezoid(prior_pdf, x=self.baseline_log_grid)
        prior_pdf /= prior_check
        assert np.min(prior_pdf) > 0, f'min prior == 0, scale={scale:.5f}'
        # convert prior PDF to PMF (pseudo-count mass function)
        return prior_pdf * self.log_dx
        
    
    def _compute_adjusted_distribution_with_laplace_prior(self, kde_pmf):
        """Compute the adjusted simulated distribution using a Laplace prior.
        Ensure that the prior is not too strong by checking the
        evaluation metrics against the specified thresholds.
        If the prior has too much influence in terms of any of the metrics tested, 
        reduce the prior strength and recompute."""

        prior_pmf = self._compute_prior_from_laplace_fit(self.location, self.scale)
        # convert the pdf to counts and apply the prior
        pseudo_counts = self.n_observations * (kde_pmf + prior_pmf * self.prior_strength)

        # re-normalize the pmf
        adjusted_pmf = pseudo_counts / np.sum(pseudo_counts)
        adjusted_pdf = adjusted_pmf / self.log_dx

        pdf_check = np.trapezoid(adjusted_pdf, x=self.baseline_log_grid)
        adjusted_pdf /= pdf_check
        
        # we are testing the prior influence, which means 
        # the kde_pmf is the "baseline_pmf" and the adjusted_pmf is the "pmf_est"
        # this will tell us how far the prior has shifted the posterior from the baseline 
        metrics = self.eval_metrics._evaluate_fdc_metrics_from_pmf(adjusted_pmf, kde_pmf)
        # check nse and kge for values LESS THAN the specified threshold

        less_than_metrics = [metrics[k] < self.eval_metrics.metric_limits[k] for k in ['nse', 'kge']]
        greater_than_metrics = [metrics[k] > self.eval_metrics.metric_limits[k] for k in ['rmse', 'relative_error', 'kld', 'emd']]
        combined_thresholds = greater_than_metrics + less_than_metrics

        if np.any(combined_thresholds):
            # find which biases are greater than 1.0e-2
            # for metric_label, bias in metrics.items():
            #     if bias > self.EvalMetrics.metric_limits[metric_label]:
            #         print(f'    Prior too strong: {metric_label} bias is > 10^-2: {bias:.2e}, adjusting prior strength to {self.prior_strength:.5e}')
            self.prior_strength *= 0.5 * self.prior_strength
            return self._compute_adjusted_distribution_with_laplace_prior(kde_pmf)
        
        return adjusted_pdf, adjusted_pmf
    

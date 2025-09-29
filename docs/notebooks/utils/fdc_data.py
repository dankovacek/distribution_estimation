from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import laplace
import json

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
        # self.eval_metrics = EvaluationMetrics(self.baseline_log_grid, self.log_w)
        # self._set_divergence_measure_functions()
        self._load_baseline_distribution()
        self._load_complete_year_dict()
        # self.location = self.ctx.laplace_param_dict['median'][stn]
        # self.scale = self.ctx.laplace_param_dict['mad'][stn]
        # self.prior_strength = self.ctx.prior_strength
        # set the measures to check for excessive influence
        self.error_check_metrics = ['mean_error', 'pct_vol_bias', 'kld', 'rmse']

        # set the maximum allowable perturbance of Q
        self.delta = context.delta


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
        # load the baseline observed PMF for the target station
        self.baseline_obs_pmf = self.ctx.baseline_obs_pmf_df[self.target_stn].values
        assert np.isclose(np.sum(self.baseline_obs_pmf), 1), f'Baseline PMF for {self.target_stn} does not sum to 1: {np.sum(self.baseline_obs_pmf)}'
        self.baseline_obs_pdf = self.ctx.baseline_obs_pdf_df[self.target_stn].values
        self.lin_x = self.ctx.baseline_obs_pdf_df.index.values
        self.log_x = np.log(self.lin_x)
        left_log_edges = self.ctx.baseline_obs_pmf_df['left_log_edges'].values
        right_log_edges = self.ctx.baseline_obs_pmf_df['right_log_edges'].values
        self.log_edges = np.concatenate(([left_log_edges[0]], right_log_edges))
        self.log_w = np.diff(self.log_edges) # widths of each bin in log space
        # assert all widths are positive and approx equal
        assert np.all(self.log_w > 0), f'Baseline PMF for {self.target_stn} has non-positive bin widths'
        
        # compute the PDF from the PMF given the linear grid (index of the pdf_df)
        pdf_area = np.trapezoid(self.baseline_obs_pdf, x=self.log_w)
        self.baseline_obs_pdf /= pdf_area  # normalize the PDF to sum to 1
        msg = f'Baseline PDF for {self.target_stn} does not sum to 1: {pdf_area}'
        assert np.isclose(np.trapezoid(self.baseline_obs_pdf, x=self.log_w), 1), msg


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
       

    def D_bits_Q_to_Qlam(self, Q, lam):
        """Exact D_bits(Q || Q_lam) for Q_lam = (1-lam)Q + lam U."""
        Q = np.asarray(Q, dtype=float)
        Q = np.clip(Q / Q.sum(), 1e-300, 1.0)
        N = Q.size
        U = 1.0 / N
        Qlam = (1.0 - lam) * Q + lam * U
        return np.sum(Q * (np.log2(Q) - np.log2(Qlam)))
    
    
    def compute_optimal_delta_limited_lambda(self, Q, maxit=100, tol=1e-6):
        """
        Largest λ ∈ [0,1] such that D_bits(Q || Q_λ) ≤ δ (exact, monotone bisection).
        """
        lo, hi = 0.0, 1.0
        if self.D_bits_Q_to_Qlam(Q, hi) <= self.delta:
            return hi
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            if self.D_bits_Q_to_Qlam(Q, mid) <= self.delta:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo
    

    def mix_with_uniform(self, Q: np.ndarray, lam: float) -> np.ndarray:
        """Shrink Q toward uniform with weight λ.  
        Applies a Dirichlet (uniform) prior to the distribution.
        lam = lambda = strength of belief in the prior
        """
        U = np.ones_like(Q) / len(Q)
        return (1.0 - lam) * Q + lam * U
    

    def _compute_adjusted_distribution_with_mixed_uniform(self, kde_pmf):
        """
        Compute a mixture Q_mixed = (1 - alpha) * Q_kde + alpha * Q_uniform
        The mixture is limited by delta, the maximum allowed perturbation between
        Q_mixture and the original Q.
        Given delta, we can compute the largest allowable lambda,
        which represents the most noise added without overly influencing Q.
        """
        lam_exact = self.compute_optimal_delta_limited_lambda(kde_pmf)
        pmf_mixed = self.mix_with_uniform(kde_pmf, lam_exact)
        pdf_mixed = pmf_mixed / self.log_w

        pdf_check = np.trapezoid(pdf_mixed, x=self.log_x)
        pdf_mixed /= pdf_check

        return pdf_mixed, pmf_mixed

    # def _compute_adjusted_distribution_with_laplace_prior(self, kde_pmf):
    #     """Compute the adjusted simulated distribution using a Laplace prior.
    #     Ensure that the prior is not too strong by checking the
    #     evaluation metrics against the specified thresholds.
    #     If the prior has too much influence in terms of any of the metrics tested, 
    #     reduce the prior strength and recompute."""

    #     prior_pmf = self._compute_prior_from_laplace_fit(self.location, self.scale)
    #     # convert the pdf to counts and apply the prior
    #     pseudo_counts = self.n_observations * (kde_pmf + prior_pmf * self.prior_strength)

    #     # re-normalize the pmf
    #     adjusted_pmf = pseudo_counts / np.sum(pseudo_counts)
    #     adjusted_pdf = adjusted_pmf / self.log_w

    #     pdf_check = np.trapezoid(adjusted_pdf, x=self.baseline_log_grid)
    #     adjusted_pdf /= pdf_check
        
    #     # we are testing the prior influence, which means 
    #     # the kde_pmf is the "baseline_pmf" and the adjusted_pmf is the "pmf_est"
    #     # this will tell us how far the prior has shifted the pdf from the baseline 
    #     metrics = self.eval_metrics._evaluate_fdc_metrics_from_pmf(adjusted_pmf, kde_pmf)
    #     # check nse and kge for values LESS THAN the specified threshold
    #     # less_than_metrics = [metrics[k] < self.eval_metrics.metric_limits[k] for k in ['nse', 'kge']]
    #     greater_than_metrics = [metrics[k] > self.eval_metrics.metric_limits[k] for k in self.error_check_metrics]
    #     combined_thresholds = greater_than_metrics #+ less_than_metrics

    #     if np.any(combined_thresholds):
    #         # find which biases are greater than 1.0e-2
    #         # for metric_label, bias in metrics.items():
    #         #     if bias > self.EvalMetrics.metric_limits[metric_label]:
    #         #         print(f'    Prior too strong: {metric_label} bias is > 10^-2: {bias:.2e}, adjusting prior strength to {self.prior_strength:.5e}')
    #         self.prior_strength *= 0.5 * self.prior_strength
    #         return self._compute_adjusted_distribution_with_laplace_prior(kde_pmf)
        
    #     return adjusted_pdf, adjusted_pmf

    # def _adjust_Q_pdf_with_prior(self, Q, label):
    #     """
    #     Adjusts the simulated PDF Q(x), originally defined on x_sim, by incorporating a Dirichlet prior,
    #     to produce an adjusted PDF Q_mod on x_target.

    #     """
    #     # Ensure the target grid lies within the global range.
    #     # Compute the global log-width.
                        
    #     # Convert Q_interp (a PDF) into effective "counts" using the number of observations.
    #     # counts_Q = N_obs * Q
        
    #     # Convert the prior density into pseudo-counts.
    #     # years_equiv = n_series * (N_obs / 365.25)
    #     # counts_prior = (self.pdf_prior / years_equiv) * dx
    #     # prior_pdf_interp = jnp.interp(self.baseline_log_grid, self.ba, prior_pdf, left=0, right=0)
    #     prior_pseudo_counts = self.knn_simulation_data[label]['prior']
    #     n_observations = self.knn_simulation_data[label]['n_obs']
        
    #     # Combine the counts from Q and the prior.
    #     Q_mod = Q * n_observations + prior_pseudo_counts
    #     # Renormalize to obtain the adjusted PDF (discrete PMF) on x_target.
    #     Q_mod /= np.sum(Q_mod)
    #     assert np.all(np.isfinite(Q_mod)), 'Q_mod is messed up'
    #     if not np.min(Q_mod) > 0:
    #         print('Q_mod min:', np.min(Q_mod))
    #         print('Q_mod sum:', np.sum(Q_mod))
    #         print('Q_mod:', Q_mod)
    #         print('Q:', Q)
    #         print('prior_pseudo_counts:', prior_pseudo_counts)
    #         print('n_observations:', n_observations)
    #         Q_mod += 1e-18
    #         Q_mod /= np.sum(Q_mod)
    #         # raise ValueError(f'Q_mod min < 0: {np.min(Q_mod)}')
    #     assert np.min(Q_mod) > 0, f'qmod_i < 0 ({np.min(Q_mod)})'
    #     assert np.isclose(np.sum(Q_mod), 1), f"Q_mod doesn't sum to 1: {np.sum(Q_mod):.5f}"

    #     q_mask = (Q > 0)
    #     prior_bias = jnp.sum(jnp.where(q_mask, Q * jnp.log2(Q / Q_mod), 0))
    #     if prior_bias < -0.0001:
    #         prior_pdf = self.knn_simulation_data[label]['prior']
    #         print('prior pmf sum =', np.sum(prior_pdf))
    #         print('Q_sum = ', np.sum(Q))
    #         print('Q_mod sum = ', np.sum(Q_mod))
    #         msg = f'    Prior bias {prior_bias:.3f} bits/sample bias'
    #         raise ValueError(msg)
        
    #     return Q_mod
    
    
    # def _compute_prior_from_laplace_fit(self, location, scale):
    #     """
    #     Fit a Laplace distribution to the simulation and define a 
    #     pdf across a pre-determined "global" range to avoid data
    #     leakage.  "Normalize" by setting the total prior mass to
    #     integrate to a factor related to the number of observations.

    #     The location of the Laplace distribution is the median, 
    #     and the scale is the mean absolute deviation (MAD).
    #     By Jensen's Inequality, the MAD is less than or equal to the standard deviation.
    #     Here we just use the predicted log-mean and log-standard deviation
    #     as an approximation of the Laplace distribution parameters.
    #     """
    #     prior_pdf = laplace.pdf(self.baseline_log_grid, loc=location, scale=scale)
    #     prior_check = np.trapezoid(prior_pdf, x=self.baseline_log_grid)
    #     prior_pdf /= prior_check
    #     assert np.min(prior_pdf) > 0, f'min prior == 0, scale={scale:.5f}'
    #     # convert prior PDF to PMF (pseudo-count mass function)
    #     return prior_pdf * self.log_w
    

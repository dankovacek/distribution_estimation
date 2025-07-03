from dataclasses import dataclass
from collections import defaultdict

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
        self.LN_param_dict = context.LN_param_dict
        self.n_grid_points = context.n_grid_points
        # self.catchments = catchments
        self.min_flow = context.min_flow # don't allow flows below this value
        self.divergence_measures = context.divergence_measures
        self.met_forcings_folder = context.LSTM_forcings_folder
        self.LSTM_ensemble_result_folder = context.LSTM_ensemble_result_folder

        self.target_da = float(self.attr_gdf[self.attr_gdf['official_id'] == stn]['drainage_area_km2'].values[0])
        self._initialize_target_streamflow_data()
        self._set_grid()
        self._set_divergence_measure_functions()
        self._load_baseline_distribution()

    
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
        df = df[df.index >= pd.to_datetime(self.ctx.daymet_start_date)]
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
    

    def _set_grid(self):        
        # self.baseline_log_grid = np.linspace(np.log(adjusted_min_uar), np.log(max_uar), self.n_grid_points)
        min_q, max_q = 1e-7, 1e4
        log_min, log_max = np.log(min_q), np.log(max_q)
        self.baseline_log_grid = np.linspace(log_min, log_max, self.n_grid_points)
        self.baseline_lin_grid = np.exp(self.baseline_log_grid)
        self.log_dx = np.gradient(self.baseline_log_grid)
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
    

    def _compute_emd(self, p, q, label=None):
        assert np.isclose(np.sum(p), 1, atol=1e-3), f'sum P = {np.sum(p)}'
        assert np.all(q >= 0), f'min q_i < 0: {np.min(q)}'
        linear_grid = np.exp(self.baseline_log_grid)
        emd = wasserstein_distance(linear_grid, linear_grid, p, q)
        return float(round(emd, 3))#, {'bias': None, 'unsupported_mass': None, 'pct_of_signal': None}

    
    def _compute_kld(self, p, q, label=None):
        # assert p and q are 1d
        assert jnp.ndim(p) >= 1, f'p is not 1D: {jnp.ndim(p)}'
        assert jnp.ndim(q) >= 1, f'q is not 1D: {jnp.ndim(q)}'
        # Ensure q is at least 2D for consistent broadcasting
        assert jnp.isclose(np.sum(p), 1, atol=1e-3), f'sum P = {np.sum(p)}'
        assert jnp.isclose(np.sum(q), 1, atol=1e-3), f'sum Q = {np.sum(q)}'
        assert jnp.all(q >= 0), f'min q_i < 0: {np.min(q)}'
        assert jnp.all(p >= 0), f'min p_i < 0: {np.min(p)}'
        p_mask = (p > 0)
        dkl = jnp.sum(jnp.where(p_mask, p * jnp.log2(p / q), 0)).item()        
        return round(dkl, 3)


    def _set_divergence_measure_functions(self):
        self.divergence_functions = {k: None for k in self.divergence_measures}
        for dm in self.divergence_measures:
            # set the divergence measure functions
            if dm == 'DKL':
                self.divergence_functions[dm] = self._compute_kld
            elif dm == 'EMD':
                self.divergence_functions[dm] = self._compute_emd
            else:
                raise Exception("only KL divergence (DKL) and Earth Mover's Distance (EMD) are implemented")
            

    def _compute_bias_from_eps(self, pmf: jnp.ndarray, divergence_measure: str, eps: float = 1e-12) -> float:
        """Compute KL divergence between original PMF and PMF + eps.

        Parameters
        ----------
        pmf : jnp.ndarray
            The original PMF (should sum to 1).
        eps : float
            Small value added to avoid zero bins.

        Returns
        -------
        float
            D_KL(pmf || pmf_eps) representing the bias introduced by smoothing.
        """
        pmf_eps = pmf + eps
        pmf_eps /= pmf_eps.sum()
        return self.divergence_functions[divergence_measure](pmf, pmf_eps)

from collections import defaultdict
from scipy.stats import wasserstein_distance
import numpy as np


class EvaluationMetrics:
    def __init__(self, data):
        """
        Initialize the EvaluationMetrics class with station data and configuration.
        
        Parameters
        ----------
        stn : str
            Station identifier.
        df : pd.DataFrame
            DataFrame containing observed discharge data.
        baseline_log_x : np.ndarray
            Log-transformed grid of flow values.
        log_w : float
            Logarithmic step size for the grid.
        """
        for k, v in data.__dict__.items():
            setattr(self, k, v)

        self.metric_limits = {
            'kld': 0.001,
            'emd': 0.05,  # this is L/s/km^2, 0.05 is very small.
            'nse': 1 - 0.001, # flipped because 1.0 is perfect
            'kge': 1 - 0.001, # flipped because 1.0 is perfect
            'mean_error': 0.01,
            'pct_vol_bias': 0.01,
            "mean_abs_rel_error": 0.01,
            'rmse': 0.01,            
        }

    def _compute_kl_divergence(self, p, q):
        """Compute the KL divergence between two probability distributions."""
        p = p.astype(float)
        q = q.astype(float)
        mask = (p > 0) & (q > 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.zeros_like(p)
            log_term[mask] = np.log(p[mask] / q[mask])
            terms = np.zeros_like(p)
            terms[mask] = p[mask] * log_term[mask]

        bad_idx = ~np.isfinite(terms)
        if np.any(bad_idx):
            print("[DEBUG] Invalid values in KL divergence computation:")
            for i in np.flatnonzero(bad_idx):
                print(
                    f"  i={i}, p={p[i]:.3e}, q={q[i]:.3e}, "
                    f"p/q={p[i]/q[i]:.3e}, log(p/q)={log_term[i]:.3e}, "
                    f"term={terms[i]:.3e}"
                )

        return np.sum(terms[mask])  # Only use valid terms
    

    def _compute_emd(self, p, q):
        assert np.isclose(np.sum(p), 1, atol=1e-3), f'sum P = {np.sum(p)}'
        assert np.all(q >= 0), f'min q_i < 0: {np.min(q)}'
        linear_grid = np.exp(self.log_x)
        emd = wasserstein_distance(linear_grid, linear_grid, p, q)
        return float(round(emd, 4))#, {'bias': None, 'unsupported_mass': None, 'pct_of_signal': None}
    

    def _compute_volumetric_pct_bias_from_fdc(self, p, q):
        """
        Compute volumetric bias between two PMFs p (observed) and q (simulated),
        defined over discrete support (flow) x.
        
        Parameters:
        - p: observed PMF (sum to 1)
        - q: simulated PMF (sum to 1)
        - x: bin centers corresponding to flow magnitudes
        
        Returns:
        - Relative bias in expected flow volume
        """
        # if p is a cdf over 1 to 99 percentiles, then compute the expected volume of p
        E_p = np.sum(p * 0.01)
        E_q = np.sum(q * 0.01)
        return (E_p - E_q) / E_p, E_p
    

    def _compute_volumetric_pct_bias_from_pmfs(self, baseline_pmf, pmf_est):
        """
        Compute volumetric bias between two PMFs p (observed) and q (simulated),
        defined over discrete support (flow) x.
        
        Parameters:
        - obs: observed flow values
        - sim: simulated flow values

        Returns:
        - Relative bias in expected flow volume
        """
        assert np.isclose(np.sum(baseline_pmf), 1), f"baseline_pmf does not sum to 1: {np.sum(baseline_pmf)}"
        assert np.isclose(np.sum(pmf_est), 1), f"pmf_est does not sum to 1: {np.sum(pmf_est)}"
        x = np.exp(self.log_x)
        E_p = np.sum(x * baseline_pmf)
        E_q = np.sum(x * pmf_est)

        return (E_p - E_q) / E_p, E_p


    def _compute_nse(self, obs, sim):
        """Compute the Nash-Sutcliffe Efficiency (NSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        # Compute the NSE
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - obs.mean()) ** 2)
        nse = 1 - (numerator / denominator)
        return nse


    def _compute_RMSE(self, obs, sim):
        """Compute the Root Mean Square Error (RMSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        # Compute the RMSE
        return np.sqrt(np.mean((obs - sim) ** 2))


    def _compute_KGE(self, obs, sim):
        """Compute the Kling-Gupta Efficiency (KGE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f"NaN values in obs: {obs}"
        assert not np.isnan(sim).any(), f"NaN values in sim: {sim}"
        if obs.size == 0:
            return np.nan
        # Compute the KGE
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = sim.mean() / obs.mean()
        beta = sim.std() / obs.std()
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        

    def _evaluate_fdc_metrics_from_pmf(self, pmf_est, baseline_obs_pmf):
        """
        Evaluate RMSE, relative error, NSE, and KGE between two FDCs represented by discrete PMFs.
        Note these are evaluated over the log_x which is set in the context.

        Parameters
        ----------
        pmf_est : np.ndarray
            Discrete PMF representing the estimated FDC, over `log_x`.
        log_x : np.ndarray
            Grid of log-transformed flow values corresponding to PMF bins.

        Returns
        -------
        dict
            Dictionary of RMSE, RelativeError, NSE, and KGE computed over p=1,...,99 quantiles.
        """
        assert (
            len(baseline_obs_pmf) == len(pmf_est) == len(self.log_x)
        ), "Array length mismatch"

        # Convert log flow grid back to linear runoff space
        # linear_grid = np.exp(self.log_x)
        log_x = self.log_x
        linear_grid = np.exp(log_x)

        # Compute CDFs
        cdf_true = np.cumsum(baseline_obs_pmf)
        cdf_true /= cdf_true[-1]
        cdf_est = np.cumsum(pmf_est)
        cdf_est /= cdf_est[-1]
        
        assert np.isfinite(cdf_true).all(), "Non-finite values in cdf_true"
        assert np.diff(cdf_true).sum() > 0, "cdf_true has no spread"

        # Percentiles (1 to 99)
        probs = np.linspace(0.01, 0.99, 99)

        # Interpolate inverse CDF (log-flow values at given probabilities)
        log_q_true = np.interp(
            probs, cdf_true, log_x, left=log_x[0], right=log_x[-1]
        )
        log_q_est = np.interp(
            probs, cdf_est, log_x, left=log_x[0], right=log_x[-1]
        )
        linear_q_true = np.interp(
            probs, cdf_true, linear_grid, left=linear_grid[0], right=linear_grid[-1]
        )
        linear_q_est = np.interp(
            probs, cdf_est, linear_grid, left=linear_grid[0], right=linear_grid[-1]
        )
        assert np.all(linear_q_true > 0), "Zero or negative values in q_true — invalid for relative error"
        assert np.all(linear_q_est > 0),  "Zero or negative values in q_est — unexpected for flow"

        # Metrics
        pct_vol_bias = np.sum((linear_q_est - linear_q_true) / np.sum(linear_q_true)) # p
        mean_error = np.mean(linear_q_est - linear_q_true) #
        mean_abs_rel_error = np.mean(np.abs(linear_q_est - linear_q_true) / linear_q_true)
        rmse = np.sqrt(np.mean((log_q_true - log_q_est) ** 2))
        nse = self._compute_nse(log_q_true, log_q_est)
        kge = self._compute_KGE(log_q_true, log_q_est)
        # volume efficiencies should be computed on linear flow values
        ve = 1 - np.sum(np.abs(linear_q_est - linear_q_true)) / np.sum(linear_q_true)
        vol_pct_bias_pmf, pmf_est_mean = self._compute_volumetric_pct_bias_from_pmfs(baseline_obs_pmf, pmf_est)
        vol_pct_bias_fdc, cdf_est_mean = self._compute_volumetric_pct_bias_from_fdc(linear_q_true, linear_q_est)
        # assert np.isclose(pmf_est_mean, cdf_est_mean, atol=0.01), f'Estimated mean Q from PMF {pmf_est_mean:.3f} does not match CDF {cdf_est_mean:.3f}'
        kld = self._compute_kl_divergence(baseline_obs_pmf, pmf_est)
        emd = self._compute_emd(baseline_obs_pmf, pmf_est)

        mean_frac_diff = (pmf_est_mean - cdf_est_mean) / pmf_est_mean
        # print(f'     PMF mean: {pmf_est_mean:.2f}, CDF mean: {cdf_est_mean:.2f} Mean frac diff: {100*mean_frac_diff:.0f}% (should be close to 0) ')
        return {
            "pct_vol_bias": float(pct_vol_bias),
            "mean_error": float(mean_error), 
            "mean_abs_rel_error": float(mean_abs_rel_error), 
            "rmse": float(rmse), 
            "nse": float(nse), 
            "kge": float(kge),
            "ve": float(ve),
            "vb_pmf": float(vol_pct_bias_pmf),
            "vb_fdc": float(vol_pct_bias_fdc),
            "kld": float(kld),
            "emd": float(emd),
            "mean_frac_diff": float(mean_frac_diff)
        }
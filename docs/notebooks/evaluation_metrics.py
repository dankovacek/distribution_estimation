from collections import defaultdict
from scipy.stats import wasserstein_distance
import numpy as np


class EvaluationMetrics:
    def __init__(self, baseline_log_grid, log_dx):
        """
        Initialize the EvaluationMetrics class with station data and configuration.
        
        Parameters
        ----------
        stn : str
            Station identifier.
        df : pd.DataFrame
            DataFrame containing observed discharge data.
        baseline_log_grid : np.ndarray
            Log-transformed grid of flow values.
        log_dx : float
            Logarithmic step size for the grid.
        """
        self.baseline_log_grid = baseline_log_grid
        self.log_dx = log_dx
        self.metric_limits = {
            'kld': 0.001,
            'emd': 0.05,  # this is L/s/km^2, 0.05 is very small.
            'nse': 1 - 0.001, # flipped because 1.0 is perfect
            'kge': 1 - 0.001, # flipped because 1.0 is perfect
            'rmse': 0.01,
            'relative_error': 0.01,
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
        linear_grid = np.exp(self.baseline_log_grid)
        emd = wasserstein_distance(linear_grid, linear_grid, p, q)
        return float(round(emd, 4))#, {'bias': None, 'unsupported_mass': None, 'pct_of_signal': None}
    

    def _compute_nse(self, obs, sim):
        """Compute the Nash-Sutcliffe Efficiency (NSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        assert (obs >= 0).all(), f'Negative values in obs: {obs}'
        assert (sim >= 0).all(), f'Negative values in sim: {sim}'
        # Compute the NSE
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - obs.mean()) ** 2)
        nse = 1 - (numerator / denominator)
        return nse


    def _compute_relative_error(self, obs, sim):
        """Compute the relative error between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        assert (obs >= 0).all(), f'Negative values in obs: {obs}'
        assert (sim >= 0).all(), f'Negative values in sim: {sim}'
        # Compute the relative error
        return (obs - sim) / obs


    def _compute_RMSE(self, obs, sim):
        """Compute the Root Mean Square Error (RMSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        assert (obs >= 0).all(), f'Negative values in obs: {obs}'
        assert (sim >= 0).all(), f'Negative values in sim: {sim}'
        # Compute the RMSE
        return np.sqrt(np.mean((obs - sim) ** 2))


    def _compute_KGE(self, obs, sim):
        """Compute the Kling-Gupta Efficiency (KGE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f"NaN values in obs: {obs}"
        assert not np.isnan(sim).any(), f"NaN values in sim: {sim}"
        assert (obs >= 0).all(), f"Negative values in obs: {obs}"
        assert (sim >= 0).all(), f"Negative values in sim: {sim}"
        if obs.size == 0:
            return np.nan
        # Compute the KGE
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = sim.mean() / obs.mean()
        beta = sim.std() / obs.std()
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


    def _evaluate_fdc_metrics_from_pmf(self, pmf_est, baseline_pmf):
        """
        Evaluate RMSE, relative error, NSE, and KGE between two FDCs represented by discrete PMFs.
        Note these are evaluated over the log_grid which is set in the context.

        Parameters
        ----------
        pmf_est : np.ndarray
            Discrete PMF representing the estimated FDC, over `log_grid`.
        log_grid : np.ndarray
            Grid of log-transformed flow values corresponding to PMF bins.

        Returns
        -------
        dict
            Dictionary of RMSE, RelativeError, NSE, and KGE computed over p=1,...,99 quantiles.
        """
        assert (
            len(baseline_pmf) == len(pmf_est) == len(self.baseline_log_grid)
        ), "Array length mismatch"

        # Convert log flow grid back to linear runoff space
        linear_grid = np.exp(self.baseline_log_grid)

        # Compute CDFs
        cdf_true = np.cumsum(baseline_pmf)
        cdf_true /= cdf_true[-1]
        cdf_est = np.cumsum(pmf_est)
        cdf_est /= cdf_est[-1]
        
        assert np.isfinite(cdf_true).all(), "Non-finite values in cdf_true"
        assert np.diff(cdf_true).sum() > 0, "cdf_true has no spread"

        # Percentiles (1 to 99)
        probs = np.linspace(0.01, 0.99, 99)

        # Interpolate inverse CDF (flow values at given probabilities)
        q_true = np.interp(
            probs, cdf_true, linear_grid, left=linear_grid[0], right=linear_grid[-1]
        )
        q_est = np.interp(
            probs, cdf_est, linear_grid, left=linear_grid[0], right=linear_grid[-1]
        )
        assert np.all(q_true > 0), "Zero or negative values in q_true — invalid for relative error"
        assert np.all(q_est > 0), "Zero or negative values in q_est — unexpected for flow"

        # Metrics
        rmse = np.sqrt(np.mean((q_true - q_est) ** 2))
        rel_error = np.mean(np.abs((q_est - q_true) / q_true))
        nse = self._compute_nse(q_true, q_est)
        kge = self._compute_KGE(q_true, q_est)

        kld = self._compute_kl_divergence(baseline_pmf, pmf_est)
        emd = self._compute_emd(baseline_pmf, pmf_est)

        return {
            "rmse": float(rmse), 
            "relative_error": float(rel_error), 
            "nse": float(nse), 
            "kge": float(kge),
            "kld": float(kld),
            "emd": float(emd),
        }


  
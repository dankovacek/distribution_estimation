# import data_processing_functions as dpf
# from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np

# ---------- Kernel Functions ----------

@jax.jit
def gaussian_kernel(u):
    return jnp.exp(-0.5 * u**2) / jnp.sqrt(2 * jnp.pi)

@jax.jit
def epanechnikov_kernel(u):
    return jnp.where(jnp.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)

@jax.jit
def top_hat_kernel(u):
    return jnp.where(jnp.abs(u) <= 1, 0.5, 0.0)

# ---------- Bandwidth Strategies ----------
def silverman_bandwidth(log_data: jnp.ndarray) -> float:
    q75, q25 = jnp.percentile(log_data, jnp.array((75, 25)))
    stdev = jnp.std(log_data)
    A = jnp.min(jnp.array([stdev, (q75 - q25) / 1.34]))
    return 1.06 * A / log_data.shape[0] ** 0.2


def measurement_error_bandwidth_function(x: jnp.ndarray) -> jnp.ndarray:
    error_points = jnp.array([1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5])
    error_values = jnp.array([1.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25])
    return jnp.interp(x, error_points, error_values, left=1.0, right=0.25)


def adaptive_bandwidths(uar: jnp.ndarray, da: float) -> jnp.ndarray:
    flow_data = uar * da / 1000
    unique_q = jnp.unique(flow_data)
    
    # compute the measurement error informed bandwidth
    # units must be volumetric flow
    error_model = measurement_error_bandwidth_function(unique_q)
    unique_UAR = (1000 / da) * unique_q
    upper_err_UAR = unique_UAR * (1 + error_model)
    err_widths_UAR = jnp.log(upper_err_UAR) - jnp.log(unique_UAR)

    # compute the basic Silverman bandwidth
    # silverman_bw = silverman_bandwidth(jnp.log(unique_UAR))

    # if there are not enough unique values, add a small amount of noise to the data
    if len(unique_UAR) < 2:
        print(f'    not enough unique values in runoff data ({len(unique_UAR)}), adding noise to the data according to the measurement error model.')
        noise_bounds = (unique_UAR * (1 - error_model), unique_UAR * (1 + error_model))
        flow_data += np.random.uniform(*noise_bounds, size=flow_data.shape)
        unique_q = jnp.unique(flow_data)
        unique_UAR = (1000 / da) * unique_q

    # compute the log midpoints and bandwidths to address the issue
    # of sparse data points in the log space
    log_midpoints = jnp.log((unique_UAR[:-1] + unique_UAR[1:]) / 2)
    left_mirror = unique_UAR[0] - (log_midpoints[0] - unique_UAR[0])
    right_mirror = unique_UAR[-1] + (unique_UAR[-1] - log_midpoints[-1])
    log_midpoints = jnp.concatenate((jnp.array([left_mirror]), log_midpoints, jnp.array([right_mirror])))
    log_diffs = jnp.diff(log_midpoints) / 2 #/ 1.15

    bw_vals = jnp.where(log_diffs > err_widths_UAR, log_diffs, err_widths_UAR)
    idx = jnp.searchsorted(unique_UAR, uar)
    return bw_vals[idx]


def kde_kernel(log_data, bw_values, log_grid):
    H = bw_values[:, None]  # (N, 1)
    U = (log_grid[None, :] - log_data) / H  # (N, M)
    K = jnp.exp(-0.5 * U**2) / (H * jnp.sqrt(2 * jnp.pi))
    return K.sum(axis=0) / log_data.shape[0]


class KDEEstimator:
    """
    Adaptive kernel density estimator using a measurement-error-informed bandwidth.

    Attributes
    ----------
    log_grid : jnp.ndarray
        Grid in log space over which to evaluate the KDE.
    dx : jnp.ndarray
        Spacing between grid points (gradient of log_grid).
    cache : dict
        Optional cache to store previously computed KDE results.
    """
    def __init__(self, log_grid, dx, cache=None):
        self.log_grid = jnp.asarray(log_grid, dtype=jnp.float32)
        self.dx = jnp.asarray(dx, dtype=jnp.float32)


    def compute(self, uar_data, da):
        """Compute the adaptive KDE and PMF for given unit area runoff data."""
        uar_data = jnp.asarray(uar_data)
        da = float(da)

        bw_values = adaptive_bandwidths(uar_data, da)
        log_data = jnp.log(uar_data)[:, None]
        pdf = kde_kernel(log_data, bw_values, self.log_grid)

        # Normalize PDF
        pdf /= jnp.trapezoid(pdf, x=self.log_grid)

        # Convert to PMF
        pmf = pdf * self.dx
        pmf /= jnp.sum(pmf)
        
        return pmf, pdf
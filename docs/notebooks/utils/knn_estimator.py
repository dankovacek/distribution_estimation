# knn_estimator.py
import numpy as np
import pandas as pd
from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor
from jax import numpy as jnp
from .kde_estimator import KDEEstimator
from time import time


class kNNEstimator:
    def __init__(self, ctx, target_stn, data, *args, **kwargs):
        self.ctx = ctx
        self.target_stn = target_stn
        self.data = data
        self.k_nearest = data.k_nearest
        # self.max_to_check_start = data.max_to_check
        # self.max_to_check = data.max_to_check
        self.weight_schemes = [1, 2] #inverse distance and inverse square distance
        self.knn_simulation_data = {}
        self.knn_pdfs = pd.DataFrame()
        self.knn_pmfs = pd.DataFrame()


    def _find_k_nearest_neighbors(self, tree_type, max_to_check):
        # Query the k+1 nearest neighbors because the first neighbor is the target point itself
        target_idx = self.ctx.id_to_idx[self.target_stn]
        if tree_type == 'spatial_dist':
            distances, indices = self.ctx.spatial_tree.query(self.ctx.coords[target_idx], k=max_to_check)
            distances *= 1 / 1000
        elif tree_type == 'attribute_dist':
            # Example query: Find the nearest neighbors for the first point
            distances, indices = self.ctx.attribute_tree.query(self.ctx.normalized_attr_values[target_idx], k=max_to_check)
        else:
            raise Exception('tree type not identified, must be one of spatial_dist, or attribute_dist.')
        
        # Remove target (self) from the results
        self_index = target_idx
        keep = indices != self_index
        indices = indices[keep]
        distances = distances[keep]

        return indices, np.round(distances, 3)
    

    def _compute_effective_k(self, df, max_k=None):
        arr = df.to_numpy()
        T, K = arr.shape
        max_k = max_k or K

        nan_mask = np.isnan(arr)
        sorted_idx = np.argsort(nan_mask, axis=1)
        row_idx = np.arange(T)[:, None]

        ks = np.arange(1, max_k + 1)
        effective_k = []
        mean_furthest = []

        for k in ks:
            idx = sorted_idx[:, :k]
            valid = ~nan_mask[row_idx, idx]
            valid_count = valid.sum(axis=1)

            effective_k.append(valid_count.mean())
            furthest_idx = np.where(valid, idx, -1).max(axis=1)
            mean_furthest.append(furthest_idx[valid_count >= k].mean() if np.any(valid_count >= k) else np.nan)

        return pd.DataFrame({
            'effective_k': np.round(effective_k, 1),
            'mean_furthest_idx': np.round(mean_furthest, 2)
        }, index=ks)
    

    def _query_distance(self, tree, id1, id2):
        """Query distance between two points in a tree using official_id."""
        if id1 not in self.ctx.id_to_idx or id2 not in self.ctx.id_to_idx:
            raise ValueError(f"One or both IDs ({id1}, {id2}) not found.")
    
        # Get indices from ID mapping
        index1, index2 = self.ctx.id_to_idx[id1], self.ctx.id_to_idx[id2]
        # Query the distance
        distance = np.linalg.norm(tree.data[index1] - tree.data[index2])  # Euclidean distance
        return distance
    

    def _process_neighbor(args):
        """
        Process a single neighbor to retrieve its data and compute the number of complete years.
        This function is designed to be used with multiprocessing.
        """
        nbr_id, dist, retrieve_fn, find_complete_fn = args

        try:
            df = retrieve_fn(nbr_id)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None

            proxy_df = df[[f'{nbr_id}_uar']]
            complete_years = list(find_complete_fn(proxy_df))
            n_years = len(complete_years)
            return (nbr_id, dist, n_years, proxy_df)
        except Exception as e:
            print(f"Failed to process {nbr_id}: {e}")
            return None
        
    
    def _retrieve_nearest_nbr_data(self, tree_type):
        MAX_CHECK = 700
        REQUIRED_GOOD = 10
        # Get the index of the target station
        
        # Query once for all potential neighbors
        nbr_idxs, dists = self._find_k_nearest_neighbors(tree_type, MAX_CHECK)
        nbr_ids = [self.ctx.idx_to_id[i] for i in nbr_idxs if self.ctx.idx_to_id[i] != self.target_stn]
        distances = [d for i, d in zip(nbr_idxs, dists) if self.ctx.idx_to_id[i] != self.target_stn]

        good_nbrs = []
        sorted_nbrs = sorted(zip(nbr_ids, distances), key=lambda x: x[1])        

        for (nbr_id, dist) in sorted_nbrs:
            df = self.data.retrieve_timeseries_discharge(nbr_id)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue  # Skip bad or empty DataFrames
            proxy_df = df[[f'{nbr_id}_uar']]
            complete_years = self.data.complete_year_dict[nbr_id]
            n_years = len(complete_years)
            good_nbrs.append((nbr_id, dist, n_years, proxy_df))
            if len(good_nbrs) == REQUIRED_GOOD:
                break

        # Concatenate the timeseries
        nbr_df = pd.concat([r[3] for r in good_nbrs], axis=1)
        # Build metadata DataFrame
        complement_type = 'attribute_dist' if tree_type == 'spatial_dist' else 'spatial_dist'
        complement_tree = getattr(self.ctx, f"{complement_type.split('_')[0]}_tree")
        scale = 1 / 1000 if complement_type == 'spatial_dist' else 1

        nbr_data = pd.DataFrame(
            [r[:3] for r in good_nbrs],
            columns=['official_id', 'distance', 'n_years']
        )
        nbr_data[complement_type] = nbr_data['official_id'].apply(
            lambda x: round(scale * self._query_distance(complement_tree, x, self.target_stn), 3)
        )

        return nbr_df, nbr_data


    def _initialize_nearest_neighbour_data(self):
        """
        Generate nearest neighbours for spatial and attribute selected k-nearest neighbours for both concurrent and asynchronous records.
        """
        print(f'    ...initializing nearest neighbours with minimum concurrent record.')
        self.nbr_dfs = defaultdict(lambda: defaultdict(dict))
        
        for tree_type in ['spatial_dist', 'attribute_dist']:
            nbr_df, nbr_data = self._retrieve_nearest_nbr_data(tree_type)
            effective_k = self._compute_effective_k(nbr_df, max_k=self.k_nearest)
            self.nbr_dfs[tree_type] = {
                'nbr_df': nbr_df,
                'nbr_data': nbr_data,
                'effective_k': effective_k,
            }
    

    def _compute_weights(self, m, k, distances, epsilon=1e-3):
        """Compute normalized inverse (square) distance weights to a given power."""

        distances = jnp.where(distances == 0, epsilon, distances)

        if k == 1:
            return jnp.array([1])
        else:
            inv_weights = 1 / (jnp.abs(distances) ** m)
            return inv_weights / jnp.sum(inv_weights)
    
      
    def _compute_frequency_ensemble_mean(self, pdfs, weights):
        """
        This function computes the weighted ensemble distribution estimates.
        """
        # Normalize distance weights
        if weights is not None:
            weights /= jnp.sum(weights).astype(jnp.float32)
            weights = jnp.array(weights)  # Ensure 1D array
            pdf_est = jnp.asarray(pdfs.to_numpy() @ weights)
        else:
            pdf_est = jnp.asarray(pdfs.mean(axis=1).to_numpy())


        # Check integral before normalization
        pdf_check = jnp.trapezoid(pdf_est, x=self.data.baseline_log_grid)
        normalized_pdf = pdf_est / pdf_check
        assert jnp.isclose(jnp.trapezoid(normalized_pdf, x=self.data.baseline_log_grid), 1), f'ensemble pdf does not integrate to 1: {pdf_check:.4f}'
                
        # Compute PMF
        pmf_est = normalized_pdf * self.data.log_dx
        pmf_est /= jnp.sum(pmf_est)

        return pmf_est, pdf_est


    # def _compute_ensemble_member_distribution_estimates(self, df):
    #     """
    #     Compute the ensemble distribution estimates based on the KNN dataframe.
    #     """    
    #     pdfs, prior_biases = pd.DataFrame(), {}
    #     # initialize a kde estimator object
    #     kde = KDEEstimator(self.data.baseline_log_grid, self.data.log_dx)
    #     for c in df.columns: 
    #         # evaluate the laplace on the prediction as a prior
    #         # drop the nan values
    #         values = df[c].dropna().values
    #         obs_count = len(values)
    #         assert len(values) > 0, f'0 values for {c}'

    #         # compute the pdf and pmf using kde
    #         assert sum(np.isnan(values)) == 0, f'NaN values in {c} {values[:5]}'

    #         kde_pmf, _ = kde.compute(
    #             values, self.data.target_da
    #         )

    #         prior = self.data._compute_prior_from_laplace_fit(values, n_cols=1) # priors are expressed in pseudo-counts
    #         # convert the pdf to counts and apply the prior
    #         counts = kde_pmf * obs_count + prior

    #         # re-normalize the pmf
    #         pmf = counts / jnp.sum(counts)
    #         pdf = pmf / self.data.log_dx

    #         pdf_check = jnp.trapezoid(pdf, x=self.data.baseline_log_grid)
    #         pdf /= pdf_check
    #         # pdf /= pdf_check
    #         assert jnp.isclose(jnp.trapezoid(pdf, x=self.data.baseline_log_grid), 1.0, atol=0.001), f'pdf does not integrate to 1 in compute_ensemble_member_distribution_estimates: {pdf_check:.4f}'
    #         pdfs[c] = pdf

    #         # convert the pdf to pmf
    #         pmf = pdf * self.data.log_dx
    #         pmf /= jnp.sum(pmf)
    #         # assert np.isclose(np.sum(pmf), 1, atol=1e-4), f'pmf does not sum to 1 in compute_ensemble_member_distribution_estimates: {np.sum(pmf):.5f}'
            
    #         # compute the bias added by the prior
    #         prior_biases[c.split('_')[0]] = {'DKL': self.data._compute_kld(kde_pmf, pmf), 'EMD': self.data._compute_emd(kde_pmf, pmf)}
    #     return pdfs, prior_biases
    
    
    def _compute_frequency_ensemble_distributions(self, nbr_df, nbr_data, distance_type):
        """
        For asynchronous comparisons, we estimate pdfs for ensemble members, then compute the mean in the time domain
        to represent the FDC simulation.  We do not do temporal averaging in this case.
        """
        knn_df_all = nbr_df.iloc[:, :self.k_nearest].copy()
        knn_data_all = nbr_data.iloc[:, :self.k_nearest].copy()
        proxy_ids = [c.split('_')[0] for c in knn_df_all.columns.tolist()]
        frequency_ensemble_pdfs = self.ctx.baseline_pmf_df[proxy_ids].copy()

        labels, pdfs, pmfs = [], [], []
        all_distances = knn_data_all['distance'].values
        all_ids = knn_data_all['official_id'].values
        # prior_bias_df = pd.DataFrame(prior_bias_dict)
        for wm in self.weight_schemes:
            for k in range(1, self.k_nearest + 1):
                distances = all_distances[:k]
                nbr_ids = all_ids[:k]
                knn_pdfs = frequency_ensemble_pdfs.iloc[:, :k].copy()

                label = f'{self.target_stn}_{k}_NN_{distance_type}_ID{wm}_freqEnsemble'
                weights = self._compute_weights(wm, k, distances)
                pmf_est, pdf_est = self._compute_frequency_ensemble_mean(knn_pdfs, weights)
                assert pmf_est is not None, f'pmf_est is None for {label}'
            
                # compute the mean number of observations (non-nan values) per row
                mean_obs_per_timestep = knn_df_all.iloc[:, :k].notna().sum(axis=1).mean()
                mean_obs_per_proxy = knn_df_all.iloc[:, :k].notna().sum(axis=0).mean()

                _, pmf_posterior = self.data._compute_posterior_with_laplace_prior(pmf_est)
                eval = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_posterior, self.data.baseline_pmf)
                bias = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_posterior, pmf_est)
      
                # compute the frequency-based ensemble pdf estimate
                self.knn_simulation_data[label] = {
                    'k': k, 'n_obs': mean_obs_per_proxy,
                    'mean_obs_per_timestep': mean_obs_per_timestep,
                    'nbrs': ','.join(nbr_ids),
                    'eval': eval,
                    'bias': bias,
                    }
                
                pdfs.append(np.asarray(pdf_est))
                pmfs.append(np.asarray(pmf_est))
                labels.append(label)

        # create a dataframe of labels(columns) for each pdf
        knn_pdfs = pd.DataFrame(pdfs, index=labels).T
        knn_pmfs = pd.DataFrame(pmfs, index=labels).T
        # Filter out already existing columns to avoid duplication
        new_pdf_cols = knn_pdfs.columns.difference(self.knn_pdfs.columns)
        new_pmf_cols = knn_pmfs.columns.difference(self.knn_pmfs.columns)
        # Concat only new columns
        self.knn_pdfs = pd.concat([self.knn_pdfs, knn_pdfs[new_pdf_cols]], axis=1)
        self.knn_pmfs = pd.concat([self.knn_pmfs, knn_pmfs[new_pmf_cols]], axis=1)
    
    
    def _delta_spike_pmf_pdf(self, single_val, log_grid):
        """
        Return a spike PMF and compatible PDF centered at the only value in the input.
        The spike is placed at the nearest log_grid point to log(single_val).
        """
        log_val = jnp.log(single_val)
        spike_idx = jnp.argmin(jnp.abs(log_grid - log_val))
        
        pmf = jnp.zeros_like(log_grid)
        pmf = pmf.at[spike_idx].set(1.0)

        dx = jnp.gradient(log_grid)
        pdf = pmf / dx  # assign all mass to one bin

        return pmf, pdf

    
    # def _compute_ensemble_contribution_metrics(self, df: pd.DataFrame, weights: np.ndarray):
    #     mask = ~df.isna()
        
    #     # Mean number of valid values per row
    #     mean_valid_per_row = mask.sum(axis=1).mean()

    #     # Normalized weights per row, masking NaNs
    #     X = df.to_numpy()
    #     W = np.broadcast_to(weights, X.shape)
    #     masked_weights = np.where(mask, W, 0.0)
    #     weight_sums = masked_weights.sum(axis=1)
    #     weight_sums[weight_sums == 0] = np.nan
    #     normalized_weights = masked_weights / weight_sums[:, None]

    #     # Average contribution per column across all rows
    #     mean_w = np.nanmean(normalized_weights, axis=0)
    #     effective_n = 1.0 / np.nansum(mean_w ** 2)

    #     return mean_valid_per_row, effective_n
    

    def _weighted_row_mean_ignore_nan(self, df: pd.DataFrame, weights: np.ndarray):
        """
        Computes the weighted mean for each row, accounting for NaNs and ensuring that
        weights are re-normalized based on valid values only. Returns a Series aligned
        with df.index, as well as ensemble stats.
        """
        X = df.to_numpy()
        mask = ~np.isnan(X)

        W = np.broadcast_to(weights, X.shape)
        masked_weights = np.where(mask, W, 0.0)

        row_weight_sums = masked_weights.sum(axis=1)
        row_weight_sums[row_weight_sums == 0] = np.nan

        normalized_weights = masked_weights / row_weight_sums[:, None]
        estimated = np.nansum(X * normalized_weights, axis=1)

        # Return as Series aligned with index
        estimated_series = pd.Series(estimated, index=df.index)

        # Also compute stats on weights
        mean_valid_per_row = mask.sum(axis=1).mean()
        mean_weight_per_col = np.nanmean(normalized_weights, axis=0)
        effective_k = 1.0 / np.nansum(mean_weight_per_col ** 2)

        return estimated_series, mean_valid_per_row, effective_k


    def _finalize_temporal_ensemble(
            self, k, label, temporal_ensemble_mean, nbrs_used,
            effective_k, mean_valid_per_row
            ):

        # Clip to prevent zero runoff issues
        temporal_ensemble_mean = np.clip(
            temporal_ensemble_mean, 1000 * 1e-4 / self.data.target_da, None
        )

        # Estimate PDF/PMF using KDE or 
        # add small amount of random noise if there is no variance
        if len(jnp.unique(temporal_ensemble_mean.values)) == 1:
            est_pmf, est_pdf = self._delta_spike_pmf_pdf(
                temporal_ensemble_mean.values[0], self.data.baseline_log_grid
            )
        else:
            est_pmf, est_pdf = self.target_kde.compute(
                temporal_ensemble_mean.values, self.data.target_da
            )

        assert est_pmf is not None, f'pmf is None for {label}'

        _, pmf_posterior = self.data._compute_posterior_with_laplace_prior(est_pmf)

        estimation_metrics = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_posterior, self.data.baseline_pmf)
        bias = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_posterior, est_pmf)

        # Store simulation outputs and metadata
        self.knn_pdfs[label] = est_pdf
        self.knn_pmfs[label] = est_pmf
        self.knn_simulation_data[label] = {
            'nbrs': nbrs_used,
            'k': k,
            'n_obs': len(temporal_ensemble_mean),
            'mean_': mean_valid_per_row,
            'mean_nbrs_per_timestep': effective_k,  # rename if clearer
            'effective_k': effective_k,
            'eval': estimation_metrics,
            'bias': bias,
        }


    def _compute_temporal_ensemble_distributions(self, distance_type, wm, nbr_df, nbr_data):
        distances = nbr_data['distance'].values
        for k in range(1, self.k_nearest + 1):
            knn_df = nbr_df.iloc[:, :k].copy()
            label = f'{self.target_stn}_{k}_NN_{distance_type}_ID{wm}_timeEnsemble'            
            weights = self._compute_weights(wm, k, distances[:k])
            temporal_ensemble_mean, mean_valid_per_row, effective_k = self._weighted_row_mean_ignore_nan(knn_df, weights)
            nbrs_used = [c.split('_')[0] for c in knn_df.columns]
            self._finalize_temporal_ensemble(
                k, label, temporal_ensemble_mean, nbrs_used,
                effective_k, mean_valid_per_row
            )
    
    
    def _compute_distribution_estimates(self, distance_type):

        nbr_df = self.nbr_dfs[distance_type]['nbr_df'].copy()
        nbr_data = self.nbr_dfs[distance_type]['nbr_data'].copy()

        for wm in self.weight_schemes:
            # compute the FDC estimate by temporal ensemble mean
            t0 = time()
            self._compute_temporal_ensemble_distributions(distance_type, wm, nbr_df, nbr_data,)
            t1 = time()
            # compute the frequency average ensemble pdfs
            self._compute_frequency_ensemble_distributions(nbr_df, nbr_data, distance_type)
            t2 = time()
            print(f'    ...{distance_type} ID{wm} took {t1 - t0:.2f}s for temporal ensemble, {t2 - t1:.2f}s for frequency ensemble.')

        # Validation
        sim_labels = list(self.knn_simulation_data.keys())
        pdf_labels = list(self.knn_pdfs.columns)
        assert set(sim_labels) == set(pdf_labels)
        
    
    def run_estimators(self):              
        self._initialize_nearest_neighbour_data()
        # set the baseline pdf by kde
        self.target_kde = KDEEstimator(self.data.baseline_log_grid, self.data.log_dx)
        for dist in ['spatial_dist', 'attribute_dist']:        
            self._compute_distribution_estimates(dist)
        return self._format_results()
    
    
    def _make_json_serializable(self, d):
        output = {}
        for k, v in d.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                output[k] = v.tolist()
            elif hasattr(v, "tolist"):
                output[k] = v.tolist()
            else:
                output[k] = v
        return output
    
    
    def _format_results(self):
        pmf_labels, pdf_labels, sim_labels = list(self.knn_pmfs.columns), list(self.knn_pdfs.columns), list(self.knn_simulation_data.keys())
        # assert label sets are the same
        assert set(pmf_labels) == set(pdf_labels), f'pmf_labels {pmf_labels} != pdf_labels {pdf_labels}'
        assert set(pmf_labels) == set(sim_labels), f'pmf_labels {pmf_labels} != sim_labels {sim_labels}'
        results = self.knn_simulation_data
        for label in pmf_labels:
            # add the pmf and pdf in a json serializable format
            results[label]['pmf'] = self.knn_pmfs[label].tolist()
            results[label]['pdf'] = self.knn_pdfs[label].tolist()
            results[label] = self._make_json_serializable(results[label])
        return results
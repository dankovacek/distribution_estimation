# knn_estimator.py
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import laplace
from jax import numpy as jnp
from .kde_estimator import KDEEstimator


class kNNFDCEstimator:
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


    def _find_k_nearest_neighbors(self, target_index, tree_type, overlapping_tree_idxs, max_to_check):
        # Query the k+1 nearest neighbors because the first neighbor is the target point itself
        if tree_type == 'spatial_dist':
            distances, indices = self.ctx.spatial_tree.query(self.ctx.coords[target_index], k=max_to_check)
            distances *= 1 / 1000
        elif tree_type == 'attribute_dist':
            # Example query: Find the nearest neighbors for the first point
            distances, indices = self.ctx.attribute_tree.query(self.ctx.normalized_attr_values[target_index], k=max_to_check)
        else:
            raise Exception('tree type not identified, must be one of spatial_dist, or attribute_dist.')
        
        # Remove target (self) from the results
        self_index = target_index
        keep = indices != self_index
        indices = indices[keep]
        distances = distances[keep]

        # Filter by the pre-processed overlapping tree indices
        if len(overlapping_tree_idxs) >= 10:
            overlap_set = set(overlapping_tree_idxs)
            filtered = [(i, d) for i, d in zip(indices, distances) if i in overlap_set]
        else:
            filtered = list(zip(indices, distances))

        neighbour_indices, neighbour_distances = zip(*filtered)
        return np.array(neighbour_indices), np.round(np.array(neighbour_distances), 3)
    

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
    

    def _find_complete_years(self, df):
        """
        Count the number of complete years in df where each of the 12 months
        has at least `min_days_per_month` valid observations.
        """
        if df.empty:
            return 0

        dates = df.dropna().index
        years = np.array([d.year for d in dates])
        months = np.array([d.month for d in dates])

        # Build count array: (year, month) → count
        ym_counts = defaultdict(int)

        for y, m in zip(years, months):
            ym_counts[(y, m)] += 1

        # Group into full years
        year_to_month_counts = defaultdict(list)
        for (y, m), count in ym_counts.items():
            year_to_month_counts[y].append(count)

        # A year is complete if it has all 12 months with enough days
        return [
            int(y) for y, counts in year_to_month_counts.items()
            if len(counts) == 12 and all(c >= self.ctx.minimum_days_per_month for c in counts)
        ]


    def _check_time_series_coverage(self, complete_proxy_years, overlap_pct):
        shared_years = list(set(complete_proxy_years) & set(self.complete_target_years))
        required_years = min(30, len(self.complete_target_years)) * int(overlap_pct) / 100.0

        if len(shared_years) >= required_years:
            return True

        print(f"    Proxy does not meet {overlap_pct}% overlap: "
            f"{len(shared_years)} shared years, requires {required_years:.0f}")
        return False

        
    def _check_neighbours(self, stn, distance, overlap_pct, target_df):
        proxy_df = self.data.retrieve_timeseries_discharge(stn)[[f'{stn}_uar']]
        if overlap_pct != '0':
            proxy_df = proxy_df.reindex(target_df.index)
        
        complete_proxy_years = self._find_complete_years(proxy_df)
        proxy_meets_coverage = self._check_time_series_coverage(complete_proxy_years, overlap_pct)
        if not proxy_meets_coverage:
            print(f'    ...skipping {stn}: <1 year or incomplete. proxy_meets_coverage={proxy_meets_coverage}, n_years={len(complete_proxy_years)}')
            return None
        return [stn, round(distance, 3), len(complete_proxy_years), proxy_df]


    def _query_distance(self, tree, id1, id2):
        """Query distance between two points in a tree using official_id."""
        if id1 not in self.ctx.id_to_idx or id2 not in self.ctx.id_to_idx:
            raise ValueError(f"One or both IDs ({id1}, {id2}) not found.")
    
        # Get indices from ID mapping
        index1, index2 = self.ctx.id_to_idx[id1], self.ctx.id_to_idx[id2]
        # Query the distance
        distance = np.linalg.norm(tree.data[index1] - tree.data[index2])  # Euclidean distance
        return distance
    

    def _retrieve_nearest_nbr_data(self, tree_type, min_overlap_proportion, tree_idx, overlapping_tree_idxs):
        MAX_CHECK = 700
        BATCH_SIZE = 10
        REQUIRED_GOOD = 10

        # Query once for all potential neighbors
        nbr_idxs, dists = self._find_k_nearest_neighbors(tree_idx, tree_type, overlapping_tree_idxs, MAX_CHECK)
        nbr_ids = [self.ctx.idx_to_id[i] for i in nbr_idxs if self.ctx.idx_to_id[i] != self.target_stn]
        distances = [d for i, d in zip(nbr_idxs, dists) if self.ctx.idx_to_id[i] != self.target_stn]

        good_nbrs = []
        sorted_nbrs = sorted(zip(nbr_ids, distances), key=lambda x: x[1])
        for (nbr_id, dist) in sorted_nbrs:
            df = self.data.retrieve_timeseries_discharge(nbr_id)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue  # Skip bad or empty DataFrames
            col = f'{nbr_id}_uar'
            if col not in df.columns:
                continue  # Skip if expected column is missing
            proxy_df = df[[col]]
            complete_years = list(self._find_complete_years(proxy_df))
            n_years = len(complete_years)
            good_nbrs.append((nbr_id, dist, n_years, proxy_df))
            if len(good_nbrs) == REQUIRED_GOOD:
                break

        while len(good_nbrs) < REQUIRED_GOOD and i < len(nbr_ids):
            batch_ids = nbr_ids[i:i + BATCH_SIZE]
            batch_dists = distances[i:i + BATCH_SIZE]
            i += BATCH_SIZE

            with ThreadPoolExecutor(max_workers=16) as executor:
                results = executor.map(self._check_neighbours,
                                    batch_ids, batch_dists,
                                    [min_overlap_proportion] * len(batch_ids),
                                    [stn_df] * len(batch_ids))

            good_nbrs.extend([r for r in results if r is not None])

        if len(good_nbrs) < REQUIRED_GOOD:
            raise Exception(f"Only {len(good_nbrs)}/{REQUIRED_GOOD} suitable neighbors found after checking {i}")

        # Sort by distance, stack timeseries
        good_nbrs.sort(key=lambda x: x[1])
        nbr_df = pd.concat([r[3] for r in good_nbrs], axis=1)

        if min_overlap_proportion != '0':
            target_series = self.data.stn_df[f'{self.target_stn}_uar'].dropna()
            nbr_df = nbr_df.reindex(target_series.index)

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


    def _contributing_ensemble_check(self, nbr_data, min_overlap, previous_ids, min_overlap_prev):
        """See how the contributing ensemble set (top ten nearest) changes as the minimum overlap proportion changes"""
        ensemble_ids = set([c.split('_')[0] for c in nbr_data.columns[:10]])
        self.n_ensemble_changes[min_overlap] = 0
        if previous_ids is not None:
            if ensemble_ids == previous_ids:
                print(f"     ....no change in neighbour set between {min_overlap_prev} and {min_overlap}% overlap.")
            else:
                added = ensemble_ids - previous_ids
                removed = previous_ids - ensemble_ids
                assert len(added) == len(removed), f'Added and removed sets should be of equal length, got {len(added)} added and {len(removed)} removed.'
                # print(f"    Change detected for min_overlap_proportion {min_overlap_prev} → {min_overlap}:")
                self.n_ensemble_changes[min_overlap] = len(added)
                # if added:
                #     print(f"    + Added:   {sorted(added)}")
                # if removed:
                #     print(f"    - Removed: {sorted(removed)}")

        previous_ids = ensemble_ids
        min_overlap_prev = min_overlap
        return previous_ids, min_overlap_prev


    def _collect_precomputed_overlap_info(self, min_overlap_proportion):
        # This function is used to collect pre-computed overlap information
        # for the target station and the given minimum overlap proportion.
        tree_idx = self.ctx.id_to_idx[self.target_stn]
        # get the pre-screened concurrent stations
        target_ws_id = str(self.ctx.official_id_dict[self.target_stn])
        # print(f'    ...collecting pre-computed overlap info for {self.target_stn} with min_overlap_proportion {min_overlap_proportion}.')
        # use the pre-computed overlap dictionary to find the concurrent stations
        overlapping_ws_ids = self.ctx.overlap_dict[min_overlap_proportion].get(str(target_ws_id), [])
        overlapping_stn_official_ids = [self.ctx.watershed_id_dict[e] for e in overlapping_ws_ids]

        existing_keys = [e for e in overlapping_stn_official_ids if e in self.ctx.id_to_idx]
        overlapping_tree_idxs = [self.ctx.id_to_idx[e] for e in existing_keys]
        overlapping_tree_idxs = [e for e in overlapping_tree_idxs if e is not None]
        assert len(overlapping_tree_idxs) >= 10, f'Not enough overlapping stations for {self.target_stn} with min_overlap_proportion {min_overlap_proportion}: {len(overlapping_tree_idxs)} found, need at least 10.'
        return tree_idx, overlapping_tree_idxs
    
    
    def _initialize_nearest_neighbour_data(self):
        """
        Generate nearest neighbours for spatial and attribute selected k-nearest neighbours for both concurrent and asynchronous records.
        """
        print(f'    ...initializing nearest neighbours with minimum concurrent record.')
        self.nbr_dfs = defaultdict(lambda: defaultdict(dict))
        
        for tree_type in ['spatial_dist', 'attribute_dist']:
            previous_ids, min_overlap_prev = None, None
            self.n_ensemble_changes = {}
            for min_overlap_proportion in self.ctx.min_target_overlap_proportions:
                min_overlap_proportion = str(min_overlap_proportion)
                tree_idx, overlapping_tree_idxs = self._collect_precomputed_overlap_info(min_overlap_proportion)
                nbr_df, nbr_data = self._retrieve_nearest_nbr_data(tree_type, min_overlap_proportion, tree_idx, overlapping_tree_idxs)
                previous_ids, min_overlap_prev = self._contributing_ensemble_check(nbr_df, min_overlap_proportion, previous_ids, min_overlap_prev)
                assert not nbr_df.empty, f'{tree_type} attr concurrent_proportion={min_overlap_proportion} nbr df empty'
                # print('       ', tree_type, concurrent, min_overlap, len(nbr_df))
                effective_k = self._compute_effective_k(nbr_df, max_k=self.k_nearest)
                self.nbr_dfs[tree_type][min_overlap_proportion] = {
                    'nbr_df': nbr_df,
                    'nbr_data': nbr_data,
                    'effective_k': effective_k,
                    'n_ensemble_changes': self.n_ensemble_changes
                }    
    

    def _compute_weights(self, m, k, distances, epsilon=1e-3):
        """Compute normalized inverse (square) distance weights to a given power."""

        distances = jnp.where(distances == 0, epsilon, distances)

        if k == 1:
            return jnp.array([1])
        else:
            inv_weights = 1 / (jnp.abs(distances) ** m)
            return inv_weights / jnp.sum(inv_weights)
    
    
    def _compute_prior_from_laplace_fit(self, predicted_uar, n_cols=1, min_prior=1e-10, scale_factor=1.05, recursion_depth=0, max_depth=100):
        """
        Fit a Laplace distribution to the simulation and define a 
        pdf across a pre-determined "global" range to avoid data
        leakage.  "Normalize" by setting the total prior mass to
        integrate to a factor related to the number of observations.
        """
        # assert no nan values
        assert np.isfinite(predicted_uar).all(), f'NaN values in predicted_uar: {predicted_uar}'
        # assert all positive values
        # assert np.all(predicted_uar > 0), f'Negative values in predicted_uar: {np.min(predicted_uar)}'
        # replace anything <= 0 with 1e-4 scaled by the drainage area
        predicted_uar = np.where(predicted_uar <= 0, 1000 * 1e-4 / self.data.target_da, predicted_uar)
        assert np.all(predicted_uar > 0), f'Negative values in predicted_uar: {np.min(predicted_uar)}'
        # print('min/max: ', np.min(predicted_uar), np.max(predicted_uar))
        loc, scale = laplace.fit(np.log(predicted_uar))       

        # Apply scale factor in case of recursion
        if scale <= 0:
            original_scale = scale
            scale = scale_factor ** recursion_depth
            print(f'   Adjusting scale from {original_scale:.3f} to {scale:.3f} for recursion depth {recursion_depth}')

        prior_pdf = laplace.pdf(self.data.baseline_log_grid, loc=loc, scale=scale)
        prior_check = jnp.trapezoid(prior_pdf, x=self.data.baseline_log_grid)
        prior_pdf /= prior_check

        # Check for zeros
        if np.any(prior_pdf == 0) | np.any(np.isnan(prior_pdf)):
            # Prevent scale from being too small
            if recursion_depth >= max_depth:
                # set a very small prior
                prior_pdf = np.ones_like(self.data.baseline_log_grid)
                err_msg = f"Recursion limit reached. Scale={scale}, setting default prior to 1 pseudo-count uniform distribution."
                print(err_msg)
                return prior_pdf
                # raise ValueError(err_msg)
            # print(f"Recursion {recursion_depth}: Zero values detected. Increasing scale to {scale:.6f}")
            return self._compute_prior_from_laplace_fit(predicted_uar, n_cols=n_cols, recursion_depth=recursion_depth + 1)
        
        second_check = jnp.trapezoid(prior_pdf, x=self.data.baseline_log_grid)
        assert np.isclose(second_check, 1, atol=2e-4), f'prior check != 1, {second_check:.6f} N={len(predicted_uar)} {predicted_uar}'
        assert np.min(prior_pdf) > 0, f'min prior == 0, scale={scale:.5f}'

        # convert prior PDF to PMF (pseudo-count mass function)
        prior_pmf = prior_pdf * self.data.log_dx

        # scale the number of pseudo-counts based on years of record  (365 / n_observations)
        # and number of models in the ensemble (given by n_cols)
        prior_pseudo_counts = prior_pmf * (365 / (len(predicted_uar) * n_cols))
        
        # return weighted_prior_pdf
        return prior_pseudo_counts
    

    def _compute_frequency_ensemble_mean(self, pdfs, weights):
        """
        This function computes the weighted ensemble distribution estimates.
        """
        # Normalize distance weights
        if weights is not None:
            weights /= jnp.sum(weights).astype(float)
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
    

    def _compute_ensemble_member_distribution_estimates(self, df):
        """
        Compute the ensemble distribution estimates based on the KNN dataframe.
        """    
        pdfs, prior_biases = pd.DataFrame(), {}
        # initialize a kde estimator object
        kde = KDEEstimator(self.data.baseline_log_grid, self.data.log_dx)
        for c in df.columns: 
            # evaluate the laplace on the prediction as a prior
            # drop the nan values
            values = df[c].dropna().values
            obs_count = len(values)
            assert len(values) > 0, f'0 values for {c}'

            # compute the pdf and pmf using kde
            assert sum(np.isnan(values)) == 0, f'NaN values in {c} {values[:5]}'

            kde_pmf, _ = kde.compute(
                values, self.data.target_da
            )

            prior = self._compute_prior_from_laplace_fit(values, n_cols=1) # priors are expressed in pseudo-counts
            # convert the pdf to counts and apply the prior
            counts = kde_pmf * obs_count + prior

            # re-normalize the pmf
            pmf = counts / jnp.sum(counts)
            pdf = pmf / self.data.log_dx

            pdf_check = jnp.trapezoid(pdf, x=self.data.baseline_log_grid)
            pdf /= pdf_check
            # pdf /= pdf_check
            assert jnp.isclose(jnp.trapezoid(pdf, x=self.data.baseline_log_grid), 1.0, atol=0.001), f'pdf does not integrate to 1 in compute_ensemble_member_distribution_estimates: {pdf_check:.4f}'
            pdfs[c] = pdf

            # convert the pdf to pmf
            pmf = pdf * self.data.log_dx
            pmf /= jnp.sum(pmf)
            # assert np.isclose(np.sum(pmf), 1, atol=1e-4), f'pmf does not sum to 1 in compute_ensemble_member_distribution_estimates: {np.sum(pmf):.5f}'
            
            # compute the bias added by the prior
            prior_biases[c.split('_')[0]] = {'DKL': self.data._compute_kld(kde_pmf, pmf), 'EMD': self.data._compute_emd(kde_pmf, pmf)}
        return pdfs, prior_biases
    
    
    def _compute_frequency_ensemble_distributions(self, nbr_df, nbr_data, distance_type, min_overlap):
        """
        For asynchronous comparisons, we estimate pdfs for ensemble members, then compute the mean in the time domain
        to represent the FDC simulation.  We do not do temporal averaging in this case.
        Default min_overlap is 0 for asynchronous comparisons.
        """
        # distances_all = knn_data_all['distance'].values[:self.k_nearest]
        # nbr_ids_all = knn_data_all['official_id'].values[:self.k_nearest]
        knn_df_all = nbr_df.iloc[:, :self.k_nearest].copy()
        knn_data_all = nbr_data.iloc[:, :self.k_nearest].copy()
        frequency_ensemble_pdfs, _ = self._compute_ensemble_member_distribution_estimates(knn_df_all)
        
        # distances = jnp.array(nbr_data['distance'].astype(float).values)
        labels, pdfs, pmfs = [], [], []
        all_distances = knn_data_all['distance'].values
        all_ids = knn_data_all['official_id'].values
        # prior_bias_df = pd.DataFrame(prior_bias_dict)
        for wm in self.weight_schemes:
            for k in range(1, self.k_nearest + 1):
                distances = all_distances[:k]
                nbr_ids = all_ids[:k]
                knn_pdfs = frequency_ensemble_pdfs.iloc[:, :k].copy()

                label = f'{self.target_stn}_{k}_NN_{min_overlap}_minOverlapPct_{distance_type}_ID{wm}_freqEnsemble'
                weights = self._compute_weights(wm, k, distances)
                pmf_est, pdf_est = self._compute_frequency_ensemble_mean(knn_pdfs, weights)
                assert pmf_est is not None, f'pmf_est is None for {label}'
            
                # compute the mean number of observations (non-nan values) per row
                mean_obs_per_timestep = knn_df_all.iloc[:, :k].notna().sum(axis=1).mean()
                mean_obs_per_proxy = knn_df_all.iloc[:, :k].notna().sum(axis=0).mean()
      
                # compute the frequency-based ensemble pdf estimate
                self.knn_simulation_data[label] = {'k': k, 'n_obs': mean_obs_per_proxy,
                                                'mean_obs_per_timestep': mean_obs_per_timestep,
                                                'nbrs': ','.join(nbr_ids)}

                self.knn_simulation_data[label]['DKL'] = self.data._compute_kld(self.ctx.baseline_pmf, pmf_est)
                self.knn_simulation_data[label]['EMD'] = self.data._compute_emd(self.ctx.baseline_pmf, pmf_est)
                
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

    
    def _compute_nse(self, obs, sim):
        """Compute the Nash-Sutcliffe Efficiency (NSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        assert (obs >= 0).all(), f'Negative values in obs: {obs}'
        assert (sim >= 0).all(), f'Negative values in sim: {sim}'
        # Compute the NSE
        numerator = jnp.sum((obs - sim) ** 2)
        denominator = jnp.sum((obs - obs.mean()) ** 2)
        nse = 1 - (numerator / denominator)
        return nse


    def _compute_KGE(self, obs, sim):
        """Compute the Kling-Gupta Efficiency (KGE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        assert (obs >= 0).all(), f'Negative values in obs: {obs}'
        assert (sim >= 0).all(), f'Negative values in sim: {sim}'
        # Compute the KGE
        r = jnp.corrcoef(obs, sim)[0, 1]
        alpha = sim.mean() / obs.mean()
        beta = sim.std() / obs.std()
        kge = 1 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return kge
    
    
    def _compute_ensemble_contribution_metrics(self, df: pd.DataFrame, weights: np.ndarray):
        mask = ~df.isna()
        
        # Mean number of valid values per row
        mean_valid_per_row = mask.sum(axis=1).mean()

        # Normalized weights per row, masking NaNs
        X = df.to_numpy()
        W = np.broadcast_to(weights, X.shape)
        masked_weights = np.where(mask, W, 0.0)
        weight_sums = masked_weights.sum(axis=1)
        weight_sums[weight_sums == 0] = np.nan
        normalized_weights = masked_weights / weight_sums[:, None]

        # Average contribution per column across all rows
        mean_w = np.nanmean(normalized_weights, axis=0)
        effective_n = 1.0 / np.nansum(mean_w ** 2)

        return mean_valid_per_row, effective_n
    
    
    def _generate_temporal_mean_ensemble_runoff_simulation(self, df, weights=None):

        assert ~df.empty, 'dataframe is empty'
        assert (df != 0).any().any(), "All values are zero in df[cols] before processing"
        
        if weights is not None:
            assert ~jnp.any(jnp.isnan(weights)), f'nan weight found: {weights}'
            assert jnp.isclose(jnp.sum(weights), 1), f'weights do not sum to 1: {weights}'
            # assert (weights > 0).all(), f'not all weights > 0, {weights}'
            assert jnp.all(weights != 0), f'Weights must not = 0: weights={weights}'
            estimated_uar = df.mul(weights, axis=1).sum(axis=1)
        else:
            assert df.isna().sum().sum() == 0, f"Some NaNs still in df: {df[df.isna().any(axis=1)]}"
            assert all(df.dtypes == 'float64') or all(np.issubdtype(t, np.floating) for t in df.dtypes), "Non-float column in df"
            estimated_uar = df.mean(axis=1, skipna=True)

        assert not np.isnan(estimated_uar).any(), "NaN values found in estimated_uar"
        assert (estimated_uar >= 0).all(), f"Estimate < 0 detected: {np.min(estimated_uar)}"
        return estimated_uar

    
    def _weighted_row_mean_ignore_nan(self, df: pd.DataFrame, weights: np.ndarray):
        """
        In the case of computing weighted means across multiple columns,
        we need to adjust the column weights to account for NaN values.
        This function computes the weighted mean for each row, ignoring NaN values.
        """
        if weights is None:
            # set the right shape for the weights
            weights = np.ones(df.shape[1], dtype=np.float64)

        assert df.shape[1] == len(weights), f"df.shape[1] != len(weights): {df.shape[1]} != {len(weights)}"

        X = df.to_numpy()
        W = np.broadcast_to(weights, X.shape)

        mask = ~np.isnan(X)  # valid entries
        masked_weights = np.where(mask, W, 0.0)

        weight_sums = masked_weights.sum(axis=1)
        # avoid division by zero
        weight_sums[weight_sums == 0] = np.nan

        normalized_weights = masked_weights / weight_sums[:, None]
        estimated = np.nansum(X * normalized_weights, axis=1)

        mean_valid_per_row, effective_k = self._compute_ensemble_contribution_metrics(df, weights)

        return pd.Series(estimated, index=df.index), mean_valid_per_row, effective_k
            

    def _get_knn_data_effective(self, k, nbr_df, nbr_data, distances, wm, distance_type, effective_k_nbrs):
        eff_k_df = self.nbr_dfs[distance_type]['concurrent']['effective_k']
        k_to_use = eff_k_df[eff_k_df['effective_k'] >= k].index[0]
        mean_furthest_idx = eff_k_df.loc[k_to_use, 'mean_furthest_idx']
        
        knn_df = nbr_df.iloc[:, :k_to_use].copy()
        weights = self._compute_weights(wm, k_to_use, distances[:k_to_use])
        temporal_ensemble_mean, mean_valid_per_row, effective_k = self._weighted_row_mean_ignore_nan(knn_df, weights)
        nbrs_used = [c.split('_')[0] for c in knn_df.columns]
        return temporal_ensemble_mean, weights, knn_df, nbrs_used, effective_k, mean_furthest_idx
    

    def _finalize_temporal_ensemble(
            self, k, label, temporal_ensemble_mean, nbrs_used,
            effective_k, mean_valid_per_row
            ):

        # Clip to prevent zero runoff issues
        temporal_ensemble_mean = np.clip(
            temporal_ensemble_mean, 1000 * 1e-4 / self.data.target_da, None
        )

        # Estimate prior from Laplace fit
        prior = self._compute_prior_from_laplace_fit(temporal_ensemble_mean, n_cols=1)

        # Compute NSE and KGE
        sim_df = temporal_ensemble_mean.rename('sim').to_frame()
        sim_obs_df = pd.concat(
            [self.data.stn_df[[f'{self.target_stn}_uar']].copy(), sim_df],
            axis=1, join='inner'
        )

        obs = sim_obs_df[f'{self.target_stn}_uar'].values
        sim = sim_obs_df['sim'].values
        nse = self._compute_nse(obs, sim)
        kge = self._compute_KGE(obs, sim)

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

        # Store simulation outputs and metadata
        self.knn_pdfs[label] = est_pdf
        self.knn_pmfs[label] = est_pmf
        self.knn_simulation_data[label] = {
            'nbrs': nbrs_used,
            'k': k,
            'prior': prior,
            'nse': nse,
            'kge': kge,
            'n_obs': len(temporal_ensemble_mean),
            'mean_': mean_valid_per_row,
            'mean_nbrs_per_timestep': effective_k,  # rename if clearer
            'effective_k': effective_k,
            'DKL': self.data._compute_kld(self.ctx.baseline_pmf, est_pmf),
            'EMD': self.data._compute_emd(self.ctx.baseline_pmf, est_pmf)
        }


    def _compute_temporal_ensemble_distributions(self, min_overlap, distance_type, wm, nbr_df, nbr_data):
        distances = nbr_data['distance'].astype(float).values
        for k in range(1, self.k_nearest + 1):
            knn_df = nbr_df.iloc[:, :k].copy()
            label = f'{self.target_stn}_{k}_NN_{min_overlap}_minOverlapPct_{distance_type}_ID{wm}_timeEnsemble'
            # handler = getattr(self, f"_get_knn_data_{k_type}")
            
            weights = self._compute_weights(wm, k, distances[:k])
            temporal_ensemble_mean, mean_valid_per_row, effective_k = self._weighted_row_mean_ignore_nan(knn_df, weights)
            nbrs_used = [c.split('_')[0] for c in knn_df.columns]

            self._finalize_temporal_ensemble(
                k, label, temporal_ensemble_mean, nbrs_used,
                effective_k, mean_valid_per_row
            )

    
    def _compute_distribution_estimates(self, distance_type):
        for min_target_overlap in self.ctx.min_target_overlap_proportions: 
            min_target_overlap = str(min_target_overlap)
            # print(f'    ...computing {distance_type} distribution estimates for min_target_overlap={min_target_overlap}')
            nbr_df = self.nbr_dfs[distance_type][min_target_overlap]['nbr_df'].copy()
            nbr_data = self.nbr_dfs[distance_type][min_target_overlap]['nbr_data'].copy()

            for wm in self.weight_schemes:
                # compute the FDC estimate by temporal ensemble mean
                self._compute_temporal_ensemble_distributions(min_target_overlap, distance_type, wm, nbr_df, nbr_data)
                # compute the frequency average ensemble pdfs
                self._compute_frequency_ensemble_distributions(nbr_df, nbr_data, distance_type, min_target_overlap)

            # Validation
            sim_labels = list(self.knn_simulation_data.keys())
            pdf_labels = list(self.knn_pdfs.columns)
            assert set(sim_labels) == set(pdf_labels)
        
    
    def run_estimators(self, divergence_measures, eps, baseline_pmf):
        self.complete_target_years = self._find_complete_years(self.data.stn_df[[f'{self.target_stn}_uar']].copy())
        # if testing asynchronous ensemble contributions from pre-180, only use the '0' overlap case
        if self.ctx.LSTM_concurrent_network is False:
             self.ctx.min_target_overlap_proportions = ['0']
                          
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
            results[label].pop('prior', None)
        for label in pmf_labels:
            # add the pmf and pdf in a json serializable format
            results[label]['pmf'] = self.knn_pmfs[label].tolist()
            results[label]['pdf'] = self.knn_pdfs[label].tolist()
            results[label] = self._make_json_serializable(results[label])
        return results
        

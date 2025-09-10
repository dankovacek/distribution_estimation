import os
import pandas as pd
import numpy as np

import jax.numpy as jnp
from scipy.stats import norm, laplace, genextreme

class ParametricFDCEstimator:
    def __init__(self, ctx, data, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.data = data
        self.target_stn = self.data.target_stn
        # self.data = data
        self.predicted_param_dict = self.ctx.predicted_param_dict
        self.predicted_param_df = pd.DataFrame(self.predicted_param_dict).T

    
    def _compute_lognorm_pmf(self, mu, sigma):
        pdf = norm.pdf(self.data.log_x, loc=mu, scale=sigma)
        pdf /= jnp.trapezoid(pdf, x=self.data.log_x)
        pmf = pdf * self.data.log_w
        pmf /= pmf.sum()
        return pmf, pdf
    

    def _compute_GEV_pmf(self, xi, mu, sigma):
        # assert values are within the valid range for GEV
        xi = max(xi, -0.5 + 1e-12)  # clip xi to avoid numerical issues
        sigma = max(sigma, 1e-12)  # ensure sigma is positive
        pdf = genextreme.pdf(self.data.log_x, xi, loc=mu, scale=sigma)
        pdf /= jnp.trapezoid(pdf, x=self.data.log_x)
        pmf = pdf * self.data.log_w
        pmf /= pmf.sum()  # normalize raw PMF
        return pmf, pdf


    def _estimate_from_mle(self):
        log_mu = self.predicted_param_dict[self.target_stn]['log_uar_mean_actual']
        log_sigma = self.predicted_param_dict[self.target_stn]['log_uar_std_actual']
        return self._compute_lognorm_pmf(log_mu, log_sigma)
    

    def _estimate_from_predicted_log_params(self):
        mu = self.predicted_param_dict[self.target_stn]['log_uar_mean_mean_predicted']
        sigma = self.predicted_param_dict[self.target_stn]['log_uar_std_mean_predicted']
        return self._compute_lognorm_pmf(mu, sigma)
        
    
    def _estimate_from_predicted_linear_mom(self):
        mean_x = self.predicted_param_dict[self.target_stn]['uar_mean_mean_predicted']
        sd_x = self.predicted_param_dict[self.target_stn]['uar_std_mean_predicted']
        v = np.log(1 + (sd_x / mean_x) ** 2)
        mu = np.log(mean_x) - 0.5 * v
        return self._compute_lognorm_pmf(mu, np.sqrt(v))
    

    def _estimate_LN_from_randomly_drawn_params(self):
        # randomly draw from the predicted parameters
        random_idx = np.random.choice(len(self.predicted_param_df))
        random_stn_idx = self.predicted_param_df.index[random_idx]
        mu_random =self.predicted_param_dict[random_stn_idx]['log_uar_mean_mean_predicted']
        sigma_random = self.predicted_param_dict[random_stn_idx]['log_uar_std_mean_predicted']
        return self._compute_lognorm_pmf(mu_random, sigma_random)


    def run_estimators(self):
        results = {}
        fns = [
            self._estimate_from_mle, 
            self._estimate_from_predicted_log_params,
            self._estimate_from_predicted_linear_mom, 
            self._estimate_LN_from_randomly_drawn_params,
            # self._estimate_from_observed_lmoments_gev,
            # self._estimate_from_predicted_lmoments_gev, 
            # self._estimate_LMOM_gev_from_randomly_drawn_params
            ]
        labels = ['MLE', 'PredictedLog', 'PredictedMOM', 'RandomDraw', 
                  #'ObsLMomentsGEV', 'PredictedLMomentsGEV', 'LMomentsGEVRandomDraw',
                  ]
        for fn, label in zip(fns, labels):
            pmf, pdf = fn()
            # _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_laplace_prior(pmf)
            _, prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf)
            if 'Moments' in label:
                # assert no nan values in the pmf
                assert not np.any(np.isnan(pmf)), f'PMF contains NaN values for {label}: {pmf[:10]}'

            results[label] = {'prior_adjusted_pmf': prior_adjusted_pmf.tolist(), 'pmf': pmf.tolist()}
            estimation_metrics = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, self.data.baseline_obs_pmf)
            results[label]['eval'] = estimation_metrics

            # compute the bias
            final_bias_metrics = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, pmf)
            results[label]['bias'] = final_bias_metrics
                
        # compute the bias from the eps
        return results
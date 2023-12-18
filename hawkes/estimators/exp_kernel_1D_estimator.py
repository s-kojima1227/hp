import numpy as np
from .log_liks.exp_kernel_1D_loglik import ExpKernel1DLogLik
from .estimator import BaseEstimator

class ExpKernel1DEstimator(BaseEstimator):
    def _build_log_lik_fn(self, events, T):
        return ExpKernel1DLogLik(events, T)

    def _get_default_init_params(self):
        return np.array([0.1, 0.1, 0.1])

    def _get_params_bounds(self):
        return np.array([[0, None], [0, None], [0, None]])
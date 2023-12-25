import numpy as np
from .loglik import LogLik
from ..base import HawkesEstimator

class Estimator(HawkesEstimator):
    def _loglik(self, events, T):
        return LogLik(events, T)

    def set_optimizer_config(self, config):
        
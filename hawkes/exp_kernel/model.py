
import numpy as np
from .loglik import ExpKernelLogLik
from .estimators import (
    ExpKernelModelEstimatorByRandomSearch,
    ExpKernelModelEstimatorByGradientMethod
)
from .simulator import ExpKernelModelSimulator

class ExpKernelModel:
    @staticmethod
    def build_estimator(type='random_search'):
        if type == 'random_search':
            return ExpKernelModelEstimatorByRandomSearch()
        elif type == 'gradient_method':
            return ExpKernelModelEstimatorByGradientMethod()
        else:
            raise ValueError('typeはrandom_searchかgradient_methodである必要があります')

    @staticmethod
    def build_simulator():
        return ExpKernelModelSimulator()

    @staticmethod
    def score(mu, a, b, events, T):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            a = np.array([[a]])
            b = np.array([[b]])
            events = [events]

        return ExpKernelLogLik(events, T)(mu, a, b)


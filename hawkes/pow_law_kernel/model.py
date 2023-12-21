import numpy as np
from .loglik import PowLawKernelLogLik
from .simulator import PowLawKernelModelSimulator
from .estimators import (
    PowLawKernelModelEstimatorByRandomSearch,
    PowLawKernelModelEstimatorByGradientMethod
)

class PowLawKernelModel:
    @staticmethod
    def build_estimator(type='random_search'):
        if type == 'random_search':
            return PowLawKernelModelEstimatorByRandomSearch()
        elif type == 'gradient_method':
            return PowLawKernelModelEstimatorByGradientMethod()
        else:
            raise ValueError('不正なtypeが指定されました')

    @staticmethod
    def build_simulator():
        return PowLawKernelModelSimulator()

    @staticmethod
    def score(mu, K, p, c, events, T):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            K = np.array([[K]])
            p = np.array([[p]])
            c = np.array([[c]])
            events = [events]

        return PowLawKernelLogLik(events, T)(mu, K, p, c)

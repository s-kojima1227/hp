from .exp import Estimator as ExpKernelEstimator, Simulator as ExpKernelSimulator
from .pow_law import Estimator as PowLawKernelEstimator, Simulator as PowLawKernelSimulator
from .sum_exp import Estimator as SumExpKernelEstimator, Simulator as SumExpKernelSimulator

class Model:
    @staticmethod
    def build_estimator(kernel):
        if kernel == 'exp':
            return ExpKernelEstimator()
        elif kernel == 'pow_law':
            return PowLawKernelEstimator()
        elif kernel == 'sum_exp':
            return SumExpKernelEstimator()
        else:
            raise ValueError('不正なカーネル名です: {}'.format(kernel))

    @staticmethod
    def build_simulator(kernel):
        if kernel == 'exp':
            return ExpKernelSimulator()
        elif kernel == 'pow_law':
            return PowLawKernelSimulator()
        elif kernel == 'sum_exp':
            return SumExpKernelSimulator()
        else:
            raise ValueError('不正なカーネル名です: {}'.format(kernel))

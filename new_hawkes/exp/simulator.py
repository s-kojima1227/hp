import numpy as np
from ..base import HawkesSimulator
from .kernel import build_kernel_matrix
from .params_converter import ParamsConverter

class Simulator:
    def __call__(self, mu, a, b, T):
        mu, a, b = ParamsConverter()(mu, a, b)

        return HawkesSimulator()(
            mu=mu,
            kernel=build_kernel_matrix(a, b),
            T=T,
            params={'mu': mu, 'a': a, 'b': b},
            kernel_type='exp',
        )

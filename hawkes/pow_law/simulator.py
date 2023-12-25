import numpy as np
from ..base import Simulator as BaseSimulator
from .kernel import build_kernel_matrix
from .converter import ParamsConverter

class Simulator(BaseSimulator):
    def __call__(self, mu, K, p, c, T):
        mu, K, p, c = ParamsConverter.toTensor(mu, K, p, c)
        return super().__call__(
            mu=mu,
            kernel=build_kernel_matrix(K, p, c),
            T=T,
            params={'mu': mu, 'K': K, 'p': p, 'c': c},
            kernel_type='pow_law'
        )

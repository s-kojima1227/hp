import numpy as np
from ..base import Simulator as BaseSimulator
from .kernel import build_kernel_matrix
from .converter import ParamsConverter

class Simulator(BaseSimulator):
    def __call__(self, mu, a, b, T):
        mu, a, b = ParamsConverter.toTensor(mu, a, b)
        return super().__call__(mu, kernel=build_kernel_matrix(a, b), T=T, params={'mu': mu, 'a': a, 'b': b}, kernel_type='exp')

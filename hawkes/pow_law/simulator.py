from ..base import Simulator as Base
from .function import Kernels
from .converter import ParamsConverter

class Simulator(Base):
    def __call__(self, mu, K, p, c, T):
        mu, K, p, c = ParamsConverter.toTensor(mu, K, p, c)
        return super().__call__(
            mu=mu,
            kernel=Kernels(K, p, c),
            T=T,
            params={'mu': mu, 'K': K, 'p': p, 'c': c},
            kernel_type='pow_law'
        )

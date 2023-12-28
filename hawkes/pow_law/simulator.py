from ..base import Simulator as Base
from .function import Kernels
from .converter import ParamsConverter as PC

class Simulator(Base):
    def __call__(self, mu, K, p, c, T):
        mu, K, p, c = PC.to_tensor(mu, K, p, c)
        return super().__call__(
            mu=mu,
            kernel=Kernels(K, p, c),
            T=T,
            params={'mu': mu, 'K': K, 'p': p, 'c': c},
            kernel_type='pow_law'
        )

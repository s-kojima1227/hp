from ..base import Simulator as Base
from .function import Kernels
from .converter import ParamsConverter

class Simulator(Base):
    def __call__(self, mu, a, b, T):
        mu, a, b = ParamsConverter.toTensor(mu, a, b)
        return super().__call__(
            mu=mu,
            kernel=Kernels(a, b),
            T=T,
            params={'mu': mu, 'a': a, 'b': b},
            kernel_type='exp'
        )

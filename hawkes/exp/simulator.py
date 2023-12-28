from ..base import Simulator as Base
from .function import Kernels
from .converter import ParamsConverter as PC

class Simulator(Base):
    def __call__(self, mu, a, b, T):
        mu, a, b = PC.to_tensor(mu, a, b)
        return super().__call__(
            mu=mu,
            kernel=Kernels(a, b),
            T=T,
            params=PC.to_dict(mu, a, b),
            kernel_type='exp'
        )

from ...base import Intensities as Base, Events
from .kernel import Kernels
from ..converter import ParamsConverter as PC

class Intensities(Base):
    def __init__(self, params, events: Events):
        mu, K, p, c = PC.unpack(params, events.dim)
        super().__init__(mu, Kernels(K, p, c), events)

from ...base import Intensities as Base, Events
from .kernel import Kernels
from ..converter import ParamsConverter

class Intensities(Base):
    def __init__(self, params, events: Events):
        mu, a, b = ParamsConverter.unpack(params, events.dim)
        super().__init__(mu, Kernels(a, b), events)

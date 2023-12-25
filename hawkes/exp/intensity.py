from ..base import Intensity as BaseIntensity
from .kernel import build_kernel_matrix

class Intensity(BaseIntensity):
    def __init__(self, mu, a, b, events):
        super().__init__(mu, build_kernel_matrix(a, b), events)

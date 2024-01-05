from ...base import Intensities as Base, Events
from .kernel import Kernels
from ..vo import Parameters as Params

class Intensities(Base):
    def __init__(self, params: Params, events: Events):
        super().__init__(
            params.baselines,
            Kernels(*params.kernel_params),
            events
        )

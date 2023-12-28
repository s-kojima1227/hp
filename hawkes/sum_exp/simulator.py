from ..base import Simulator as Base
from .function import Kernels
from .converter import ParamsConverter as PC

class Simulator(Base):
    def __call__(self, baselines, adjacencies, decays, end_time):
        return super().__call__(
            baselines,
            Kernels(adjacencies, decays),
            end_time,
            params=PC.to_dict(baselines, adjacencies, decays),
            kernel_type='sum_exp'
        )

from ..base import Simulator as Base
from .function import Kernels
from .vo import Parameters as Params, ParametersFactory as PF

class Simulator(Base):
    def set_params(self, baselines, multipliers, exponents, cutoffs):
        self.__params = PF().build_from_packed(baselines, (multipliers, exponents, cutoffs))

    @property
    def _params(self) -> Params:
        return self.__params

    @property
    def _kernels(self) -> Kernels:
        return Kernels(*self.__params.kernel_params)

    @property
    def _kernel_type(self) -> str:
        return 'pow_law'

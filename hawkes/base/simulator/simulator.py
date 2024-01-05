import numpy as np
from .thinning_method import ThinningMethod
from .output import Output
from ..function import Intensities, Kernels
from ..vo import Parameters as Params
from abc import ABC, abstractmethod

class Simulator(ABC):
    def __init__(self):
        self._events = None
        self._intensities = None
        self._end_time = None

    def __call__(self, end_time) -> Output:
        self._end_time = end_time
        self._simulate()
        self._calc_intensities()
        return self._build_output()

    def _simulate(self) -> None:
        self._events = ThinningMethod(self._params.baselines, self._kernels)(self._end_time)

    def _calc_intensities(self) -> None:
        t = np.arange(0, self._end_time + 1, 1)
        intensities_fn = Intensities(
            self._params.baselines,
            self._kernels,
            self._events
        )
        self._intensities = t, intensities_fn(t)

    def _build_output(self) -> Output:
        return Output(
            events=self._events,
            intensity=self._intensities,
            params=self._params,
            kernel_type=self._kernel_type
        )

    @property
    @abstractmethod
    def _params(self) -> Params:
        pass

    @property
    @abstractmethod
    def _kernels(self) -> Kernels:
        pass

    @property
    @abstractmethod
    def _kernel_type(self) -> str:
        pass

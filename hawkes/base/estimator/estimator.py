from abc import ABC, abstractmethod
import numpy as np
from .minimizer import MinimizerFactory
from .output import Output
from ..vo import Events, EventsFactory as EF, Parameters as Params
from ..function import Loss, Intensities
from typing import Dict

class Estimator(ABC):
    def __init__(self):
        self.__minimization_config = None
        self.__minimizer = None
        self.__dim = None
        self.__events = None
        self.__params = None
        self.__score = None
        self.__intensities = None

    def __call__(self, events, end_time) -> Output:
        self._set_events(events, end_time)
        self._build_minimizer()
        self._estimate()
        self._calc_intensities()
        return self._build_output()

    def _set_events(self, events, end_time) -> None:
        if isinstance(events, np.ndarray):
            events = [events]
        self.__events =  EF.from_events_grouped_by_mark(events, end_time)
        self.__dim = self.__events.dim

    def _estimate(self) -> None:
        self.__params, self.__score = self.__minimizer(self._loss_fn)

    def _calc_intensities(self) -> None:
        t = np.arange(0, self.__events.end_time + 1, 1)
        self.__intensities = (t, self._intensities_fn(t))

    def _build_output(self) -> Output:
        return Output(
            events=self.__events,
            intensity=self.__intensities,
            params=self._params_vo,
            kernel_type=self._kernel_type,
            loglik=-self.__score,
        )

    def _build_minimizer(self):
        minimization_config = self.__minimization_config
        if minimization_config is None:
            minimization_config = self._default_minimization_config
        self.__minimizer = (MinimizerFactory())(minimization_config)

    def set_minimization_config(self, method, option):
        self.__minimization_config = {
            'method': method,
            'option': option,
        }

    @property
    def _dim(self) -> int:
        return self.__dim

    @property
    def _events(self) -> Events:
        return self.__events

    @property
    def _params(self) -> np.ndarray:
        return self.__params

    @property
    def _score(self) -> float:
        return self.__score

    @property
    @abstractmethod
    def _loss_fn(self) -> Loss:
        pass

    @property
    @abstractmethod
    def _intensities_fn(self) -> Intensities:
        pass

    @property
    @abstractmethod
    def _params_vo(self) -> Params:
        pass

    @property
    @abstractmethod
    def _default_minimization_config(self) -> Dict:
        pass

    @property
    @abstractmethod
    def _kernel_type(self) -> str:
        pass

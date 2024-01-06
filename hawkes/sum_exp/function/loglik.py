from ...base import LogLik as Base, Events
from .intensity import Intensities
from .compensator import compensators
from ..vo import ParametersFactory as PF

class LogLik(Base):
    def _intensity_i(self, mark, time, events: Events, params):
        params = PF(events.dim, self._num_exps).build_from_unpacked(params)
        intensities = Intensities(params, events)
        return intensities[mark](time)

    def _compensators(self, time, events: Events, params):
        params = PF(events.dim, self._num_exps).build_from_unpacked(params)
        return compensators(time, events, params)

    def grad(self, params):
        raise NotImplementedError

    def set_num_exps(self, num_exps):
        self._num_exps = num_exps

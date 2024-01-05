from ...base import Events
from ..vo import Parameters as Params
import numpy as np

# FIXME: クラス化
def compensators(t, events: Events, params: Params) -> np.ndarray:
    H_T = events.ordered_by_time
    H_t = H_T[H_T[:, 0] < t]
    baselines = params.baselines
    adjacencies, decays = params.kernel_params

    compensators = baselines * t

    for (t_i, m_i) in H_t:
        m_i = int(m_i)
        compensators += adjacencies[m_i] * (1 - np.exp(-decays[m_i] * (t - t_i)))

    return compensators

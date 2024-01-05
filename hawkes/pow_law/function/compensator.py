from ...base import Events
from ..vo import Parameters as Params
import numpy as np

# FIXME: クラス化
def compensators(t, events: Events, params: Params) -> np.ndarray:
    H_T = events.ordered_by_time
    H_t = H_T[H_T[:, 0] < t]
    baselines = params.baselines
    multipliers, exponents, cutoffs = params.kernel_params

    compensators = baselines * t

    for (t_i, m_i) in H_t:
        m_i = int(m_i)
        compensators -= \
            (multipliers[m_i] / (exponents[m_i] - 1)) * \
            ((1 / (t - t_i + cutoffs[m_i])) ** (exponents[m_i] - 1) - \
            (1 / cutoffs[m_i]) ** (exponents[m_i] - 1))

    return compensators

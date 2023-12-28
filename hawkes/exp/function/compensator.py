from ...base import Events
from ..converter import ParamsConverter as PC
import numpy as np

# FIXME: クラス化する？
def compensators(t, events: Events, params) -> np.ndarray:
    H_T = events.ordered_by_time
    H_t = H_T[H_T[:, 0] < t]
    mu, a, b = PC.unpack(params, events.dim)

    compensators = mu * t

    for (t_i, m_i) in H_t:
        m_i = int(m_i)
        compensators += a[m_i] * (1 - np.exp(-b[m_i] * (t - t_i)))

    return compensators

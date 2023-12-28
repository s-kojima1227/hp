from ...base import Events
from ..converter import ParamsConverter
import numpy as np

# FIXME: クラス化する？
def compensators(t, events: Events, params):
    H_T = events.ordered_by_time
    mu, a, b = ParamsConverter.unpack(params, events.dim)

    compensators = mu * t

    for (t_i, m_i) in H_T:
        m_i = int(m_i)
        compensators += a[m_i] * (1 - np.exp(-b[m_i] * (t - t_i)))

    return compensators

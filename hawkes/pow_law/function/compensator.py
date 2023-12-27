from ...base import Events
from ..converter import ParamsConverter

# FIXME: クラス化する？
def compensators(t, events: Events, params):
    H_T = events.ordered_by_time
    mu, K, p, c = ParamsConverter.unpack(params, events.dim)

    compensators = mu * t

    for (t_i, m_i) in H_T:
        m_i = int(m_i)
        compensators -= (K[m_i] / (p[m_i] - 1)) * ((1 / (t - t_i + c[m_i])) ** (p[m_i] - 1) - (1 / c[m_i]) ** (p[m_i] - 1))

    return compensators

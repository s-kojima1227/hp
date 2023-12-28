from ...base import Events
from ..converter import ParamsConverter as PC

# FIXME: クラス化する？
def compensators(t, events: Events, params):
    H_T = events.ordered_by_time
    H_t = H_T[H_T[:, 0] < t]
    mu, K, p, c = PC.unpack(params, events.dim)

    compensators = mu * t

    for (t_i, m_i) in H_t:
        m_i = int(m_i)
        compensators -= (K[m_i] / (p[m_i] - 1)) * ((1 / (t - t_i + c[m_i])) ** (p[m_i] - 1) - (1 / c[m_i]) ** (p[m_i] - 1))

    return compensators

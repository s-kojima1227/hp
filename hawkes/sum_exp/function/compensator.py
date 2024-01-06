from ...base import Events
from ..vo import Parameters as Params
import numpy as np

def compensators(t: float, events: Events, params: Params) -> np.ndarray:
    H_T = events.ordered_by_time
    H_t = H_T[H_T[:, 0] < t]
    baselines = params.baselines
    adjacencies, decays = params.kernel_params

    dim = events.dim
    num_exps = params.num_exps

    compensators = np.zeros(dim)

    for i in range(dim):
        compensators_i = baselines[i] * t
        for t_j, m_j in H_t:
            for u in range(num_exps):
                compensators_i += adjacencies[i, int(m_j-1), u] * (1 - np.exp(-decays[u] * (t - t_j)))
        compensators[i] = compensators_i
    return compensators

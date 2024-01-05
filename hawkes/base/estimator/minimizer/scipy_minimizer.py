from scipy.optimize import minimize
from .minimizer import Minimizer as Base
from ...vo import Parameters as Params

class ScipyMinimizer(Base):
    def __init__(self, init_params: Params, bounds):
        self._init_params = init_params.unpacked
        self._bounds = bounds

    def __call__(self, objective_fn):
        result = minimize(objective_fn, self._init_params, bounds=self._bounds)
        return result.x, result.fun

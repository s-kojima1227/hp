import numpy as np
from scipy.optimize import minimize

class ScipyMinimizer:
    def __init__(self, init_params, bounds):
        self._init_params = init_params
        self._bounds = bounds

    def __call__(self, objective_fn, verbose=True):
        result = minimize(objective_fn, self._init_params, bounds=self._bounds)
        return result.x, result.fun

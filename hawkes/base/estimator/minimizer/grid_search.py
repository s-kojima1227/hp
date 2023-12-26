import numpy as np
from scipy.optimize import brute

class GridSearch:
    def __init__(self, grid):
        self._grid = grid

    def __call__(self, objective_fn, verbose=True):
        x0, fval, grid, Jout = brute(objective_fn, self._grid, finish=None, full_output=True)
        return x0, fval

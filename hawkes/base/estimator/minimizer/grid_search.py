import numpy as np
from scipy.optimize import brute
from .minimizer import Minimizer as Base

class GridSearch(Base):
    def __init__(self, grid):
        self._grid = grid

    def __call__(self, objective_fn):
        x0, fval, grid, Jout = brute(objective_fn, self._grid, finish=None, full_output=True)
        return x0, fval

import numpy as np
from .minimizer import Minimizer as Base

class RandomSearch(Base):
    def __init__(self, n_iter, bounds):
        self._n_iter = n_iter
        self._bounds = bounds

    def __call__(self, objective_func):
        best_score = np.inf
        best_params = None

        for i in range(self._n_iter):
            params = np.array([np.random.uniform(low, high) for low, high in self._bounds])
            score = objective_func(params)

            if score < best_score:
                print('best score updated: iter={}, score={:.3f}, params={}'.format(i, score, params))
                best_score = score
                best_params = params

            if i % 1000 == 0:
                print('iter: {}'.format(i))

        return best_params, best_score

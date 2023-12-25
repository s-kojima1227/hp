import numpy as np

class RandomSearch:
    def __init__(self, n_iter, search_space):
        self._n_iter = n_iter
        self._search_space = search_space

    def __call__(self, objective_func, verbose=True):
        best_score = -np.inf
        best_params = None

        for i in range(self._n_iter):
            params = np.array([np.random.uniform(low, high) for low, high in self._search_space])
            score = objective_func(params)

            if score > best_score:
                if verbose:
                    print('best score updated: iter={}, score={:.3f}, params={}'.format(i, score, params))
                best_score = score
                best_params = params

            if i % 1000 == 0 and verbose:
                print('iter: {}'.format(i))

        return best_params, best_score

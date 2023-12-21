import numpy as np

class RandomSearchOptimizer:
    def __init__(self, n_iter=10000, search_space=None):
        self._n_iter = n_iter
        self._search_space = search_space

    def __call__(self, objective_func):
        best_score = -np.inf
        best_params = None

        for i in range(self._n_iter):
            params = {k: self._generate_random_params(bounds) for k, bounds in self._search_space.items()}
            score = objective_func(**params)

            if score > best_score:
                print('best score updated: iter={}, score={:.3f}, params={}'.format(i, score, params))
                best_score = score
                best_params = params
            if i % 1000 == 0:
                print('iter: {}'.format(i))

        return best_params, best_score

    def _generate_random_params(self, bounds):
        """
        パラメータの形状と範囲に基づいてパラメータ値をランダム生成する
        """
        if np.array(bounds).ndim == 1:
            # スカラーパラメータ
            low, high = bounds
            return [np.random.uniform(low, high)]
        elif np.array(bounds).ndim == 2:
            # ベクトルパラメータ
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        elif np.array(bounds).ndim == 3:
            # 行列パラメータ
            return np.array([[np.random.uniform(low, high) for low, high in row_bounds] for row_bounds in bounds])

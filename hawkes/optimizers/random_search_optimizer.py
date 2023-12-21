import numpy as np

class RandomSearchOptimizerSettings:
    def __init__(self, n_iter=10000, search_space=None):
        self._n_iter = n_iter
        self._search_space = search_space

    @property
    def n_iter(self):
        return self._n_iter

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        self._search_space = search_space

class RandomSearchOptimizer:
    def __init__(self, settings: RandomSearchOptimizerSettings):
        self._n_iter = settings.n_iter
        self._search_space = settings.search_space

    def __call__(self, objective_func):
        """
        ランダムサーチを使用して、目的関数を最大化するパラメータを見つけます。
        :param objective_func: 目的関数
        :param param_bounds: パラメータの範囲
        :param n_iterations: イテレーションの回数
        """
        best_score = -np.inf
        best_params = None

        for _ in range(self._n_iter):
            params = {k: self._generate_random_params(bounds) for k, bounds in self._search_space.items()}
            score = objective_func(**params)

            if score > best_score:
                print('回数: {}'.format(_))
                print('ベストスコア更新: {:.3f}'.format(score))
                print('パラメータ: {}'.format(params))
                best_score = score
                best_params = params

        return best_params, best_score

    def _generate_random_params(self, bounds):
        """
        パラメータの形状と範囲に基づいてパラメータ値をランダム生成する
        """
        if np.array(bounds).ndim == 1:
            # スカラーパラメータ
            low, high = bounds
            return np.random.uniform(low, high)
        elif np.array(bounds).ndim == 2:
            # ベクトルパラメータ
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        elif np.array(bounds).ndim == 3:
            # 行列パラメータ
            return np.array([[np.random.uniform(low, high) for low, high in row_bounds] for row_bounds in bounds])

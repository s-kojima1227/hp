import numpy as np

class GradientOptimizerSettings:
    def __init__(self, learning_rate=0.00001, max_iter=1000000, tol=0.001, init_params=None):
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._tol = tol
        self._init_params = init_params

    def set_params_order(self, params_order):
        self._params_order = params_order

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def tol(self):
        return self._tol

    @property
    def init_params(self):
        if (self._init_params is None) or (self._params_order is None):
            raise ValueError('初期パラメータ、またはパラメータの順番が設定されていません')
        return np.array([self._init_params[key] for key in self._params_order])

    @init_params.setter
    def init_params(self, init_params):
        self._init_params = init_params


    @property
    def is_init_param_set(self):
        return self._init_params is not None

class GradientOptimizer:
    def __init__(self, settings: GradientOptimizerSettings):
        self._init_params = settings.init_params
        self._learning_rate = settings.learning_rate
        self._max_iter = settings.max_iter
        self._tol = settings.tol

    def __call__(self, objective_fn):
        params = self._init_params

        for i in range(self._max_iter):
            grad = objective_fn.grad(params)
            params += self._learning_rate * grad

            if i % 1000 == 0:
                print('iter: {}, params: {}, grad: {}'.format(i, params, grad))

            if np.linalg.norm(grad) < self._tol:
                break

        return params, objective_fn(*params)

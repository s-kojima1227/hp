import numpy as np
from .minimizer import Minimizer as Base
from ...vo import Parameters as Params

class GradientMethod(Base):
    def __init__(self, learning_rate, max_iter, tol, init_params: Params):
        self._init_params: Params = init_params
        self._learning_rate = learning_rate
        self._max_iter = int(max_iter)
        self._tol = tol

    def __call__(self, objective_fn):
        params = self._init_params.unpacked

        for i in range(self._max_iter):
            grad = objective_fn.grad(params.unpacked)
            params -= self._learning_rate * grad
            grad_norm = np.linalg.norm(grad)

            if i % 1000 == 0:
                print('iter: {}, params: {}, grad: {}, grad_norm: {}'.format(i, params, grad, grad_norm))

            if grad_norm < self._tol:
                break

        return params, objective_fn(params)

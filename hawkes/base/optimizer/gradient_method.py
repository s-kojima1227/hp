import numpy as np

class GradientMethodOptimizer:
    def __init__(self, learning_rate=0.00001, max_iter=1000000, tol=0.001, init_params=None):
        self._init_params = init_params
        self._learning_rate = learning_rate
        self._max_iter = int(max_iter)
        self._tol = tol

    def __call__(self, objective_fn):
        params = self._init_params

        for i in range(self._max_iter):
            grad = objective_fn.grad(params)
            params += self._learning_rate * grad
            grad_norm = np.linalg.norm(grad)

            if i % 1000 == 0:
                print('iter: {}, params: {}, grad: {}, grad_norm: {}'.format(i, params, grad, grad_norm))

            if grad_norm < self._tol:
                break

        return params

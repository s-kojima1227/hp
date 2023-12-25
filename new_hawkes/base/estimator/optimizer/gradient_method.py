import numpy as np

class GradientMethod:
    def __init__(self, learning_rate, max_iter, tol, init_params):
        self._init_params = init_params
        self._learning_rate = learning_rate
        self._max_iter = int(max_iter)
        self._tol = tol

    def __call__(self, objective_fn, verbose=True):
        params = self._init_params

        for i in range(self._max_iter):
            grad = objective_fn.grad(params)
            params += self._learning_rate * grad
            grad_norm = np.linalg.norm(grad)

            if i % 1000 == 0 and verbose:
                print('iter: {}, params: {}, grad: {}, grad_norm: {}'.format(i, params, grad, grad_norm))

            if grad_norm < self._tol:
                break

        return params, objective_fn(params)

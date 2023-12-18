import numpy as np
from .optimizer import Optimizer

class GradientOptimizer(Optimizer):
    def __init__(self, learning_rate=0.00001, max_iter=1000000, tol=0.001):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def __call__(self, objective_fn, init_params, bounds=None):
        params = init_params
        for i in range(self.max_iter):
            grad = objective_fn.grad(params)
            params += self.learning_rate * grad

            # パラメータが定義域を超えないようにクリッピング
            # if bounds is not None:
            #     params = np.array([np.clip(param, bounds[i][0], bounds[i][1]) for i, param in enumerate(params)])

            if i % 1000 == 0:
                print('iter: {}, params: {}, grad: {}'.format(i, params, grad))

            if np.linalg.norm(grad) < self.tol:
                break

        return params

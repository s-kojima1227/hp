import numpy as np

class Kernel:
    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b

    def __call__(self, t):
        return self._a * self._b * np.exp(-self._b * t)

def build_kernel_matrix(a: np.ndarray, b: np.ndarray):
    if not (a.shape == b.shape):
        raise ValueError('パラメーターの次元が不適切です: a.shape={}, b.shape={}'.format(a.shape, b.shape))
    dim = a.shape[0]
    return np.array([[Kernel(a[i, j], b[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)

import numpy as np

class ParamsConverter:
    def __call__(self, mu, a, b):
        if isinstance(mu, (int, float)):
            return (np.array([mu]), np.array([[a]]), np.array([[b]]))
        else:
            return (mu, a, b)

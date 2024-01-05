import numpy as np

class BoundsConverter:
    @staticmethod
    def to_tensor(bounds_mu, bounds_a, bounds_b):
        if isinstance(bounds_mu, (tuple, slice)):
            return (np.array([bounds_mu]), np.array([[bounds_a]]), np.array([[bounds_b]]))
        else:
            return (np.array(bounds_mu), np.array(bounds_a), np.array(bounds_b))

    @staticmethod
    def pack(bounds_mu, bounds_a, bounds_b):
        bounds_a = np.array([bounds_a_ij for bounds_a_i in bounds_a for bounds_a_ij in bounds_a_i])
        bounds_b = np.array([bounds_b_ij for bounds_b_i in bounds_b for bounds_b_ij in bounds_b_i])
        return np.concatenate([bounds_mu, bounds_a, bounds_b])

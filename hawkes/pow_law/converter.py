import numpy as np

class BoundsConverter:
    @staticmethod
    def to_tensor(bounds_mu, bounds_K, bounds_p, bounds_c):
        if isinstance(bounds_mu, (tuple, slice)):
            return (np.array([bounds_mu]), np.array([[bounds_K]]), np.array([[bounds_p]]), np.array([[bounds_c]]))
        else:
            return (np.array(bounds_mu), np.array(bounds_K), np.array(bounds_p), np.array(bounds_c))

    @staticmethod
    def pack(bounds_mu, bounds_K, bounds_p, bounds_c):
        bounds_K = np.array([bounds_K_ij for bounds_K_i in bounds_K for bounds_K_ij in bounds_K_i])
        bounds_p = np.array([bounds_p_ij for bounds_p_i in bounds_p for bounds_p_ij in bounds_p_i])
        bounds_c = np.array([bounds_c_ij for bounds_c_i in bounds_c for bounds_c_ij in bounds_c_i])
        return np.concatenate([bounds_mu, bounds_K, bounds_p, bounds_c])

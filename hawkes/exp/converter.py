import numpy as np

class ParamsConverter:
    @staticmethod
    def toTensor(mu, a, b):
        if isinstance(mu, (int, float)):
            return (np.array([mu]), np.array([[a]]), np.array([[b]]))
        else:
            return (np.array(mu), np.array(a), np.array(b))

    @staticmethod
    def pack(mu, a, b):
        return np.concatenate([mu, a.flatten(), b.flatten()])

    @staticmethod
    def unpack(params, dim):
        mu = params[:dim]
        a = params[dim:dim * (dim + 1)].reshape(dim, dim)
        b = params[dim * (dim + 1):].reshape(dim, dim)
        return mu, a, b

    def toDict(mu, a, b):
        return {'mu': mu, 'a': a, 'b': b}

class BoundsConverter:
    @staticmethod
    def toTensor(bounds_mu, bounds_a, bounds_b):
        if isinstance(bounds_mu, (tuple, slice)):
            return (np.array([bounds_mu]), np.array([[bounds_a]]), np.array([[bounds_b]]))
        else:
            return (np.array(bounds_mu), np.array(bounds_a), np.array(bounds_b))

    @staticmethod
    def pack(bounds_mu, bounds_a, bounds_b):
        bounds_a = np.array([bounds_a_ij for bounds_a_i in bounds_a for bounds_a_ij in bounds_a_i])
        bounds_b = np.array([bounds_b_ij for bounds_b_i in bounds_b for bounds_b_ij in bounds_b_i])
        return np.concatenate([bounds_mu, bounds_a, bounds_b])

import numpy as np

class ParamsConverter:
    @staticmethod
    def toTensor(mu, K, p, c):
        if isinstance(mu, (int, float)):
            return (np.array([mu]), np.array([[K]]), np.array([[p]]), np.array([[c]]))
        else:
            return (np.array(mu), np.array(K), np.array(p), np.array(c))

    @staticmethod
    def pack(mu, K, p, c):
        return np.concatenate([mu, K.flatten(), p.flatten(), c.flatten()])

    @staticmethod
    def unpack(params, dim):
        mu = params[:dim]
        K = params[dim:dim * (dim + 1)].reshape(dim, dim)
        p = params[dim * (dim + 1):dim * (2 * dim + 1)].reshape(dim, dim)
        c = params[dim * (2 * dim + 1):].reshape(dim, dim)
        return mu, K, p, c

    def toDict(mu, K, p, c):
        return {'mu': mu, 'K': K, 'p': p, 'c': c}

class BoundsConverter:
    @staticmethod
    def toTensor(bounds_mu, bounds_K, bounds_p, bounds_c):
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

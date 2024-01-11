import numpy as np
from ..base import Events, Kernels


class BaselinesParameters:
    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

class KernelParameters:
    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.calc(x)
        return Variable(y)

    def calc(self, input):
        raise NotImplementedError()

class Baselines(Function):
    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

    def calc(self, input):
        return np.sum(self.alphas * np.exp(-self.betas * input), axis=1)

class Kernel(Function):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def calc(self, input):
        return self.alpha * np.exp(-self.beta * input)

class KernelMatrix:
    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas
        self.value = np.array([[Kernel(alphas[i, j], betas[i, j]) for j in range(alphas.shape[1])] for i in range(alphas.shape[0])])

    def __getitem__(self, i):
        return self.value[i]

class Compensators(Function):
    def __init__(self, baselines: Baselines, kernel_matrix: KernelMatrix, events: Events):
        self.baselines = baselines
        self.kernel_matrix = kernel_matrix
        self.events = events

    def calc(self, input):
        H_T = self.events.ordered_by_time
        H_t = H_T[H_T[:, 0] < input]
        dim = self.events.dim

        def compensator_i(t, i):
            compensator_i = self.baselines[i](t) * t
            for (t_i, m_i) in H_t:
                m_i = int(m_i)
                compensator_i += self.kernel_matrix[i][m_i](t - t_i) - self.kernel_matrix[i][m_i](0)
            return compensator_i

        return np.array([compensator_i(input, i) for i in range(dim)])

class Intensities(Function):
    def __init__(self, baselines: Baselines, kernel_matrix: KernelMatrix, events: Events):
        self.baselines = baselines
        self.kernel_matrix = kernel_matrix
        self.events = events
        self.dim = events.dim

    def calc(self, input):
        H_T = self.events.ordered_by_time
        H_t = H_T[H_T[:, 0] < input]

        def intensity_i(t, i):
            intensity_i = self.baselines[i](t)
            for (t_i, m_i) in H_t:
                m_i = int(m_i)
                intensity_i += self.kernel_matrix[i][m_i](t - t_i)
            return intensity_i

        return np.array([intensity_i(input, i) for i in range(self.dim)])

class LogLik(Function):
    def __init__(self, events: Events, baselines: Baselines, kernel_matrix: KernelMatrix):
        self.events = events
        self.baselines = baselines
        self.kernel_matrix = kernel_matrix
        self.compensators = Compensators(baselines, kernel_matrix, events)

    def calc(self, input):
        H_T = self.events.ordered_by_time
        T = self.events.end_time

        loglik = 0
        for (t_i, m_i) in H_T:
            intensity = 
            intensity = self.intensity_i(int(m_i), t_i)
            loglik += np.log(intensity)

        compensators = self.compensators(T)
        loglik -= np.sum(compensators)

        return loglik
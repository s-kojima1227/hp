import numpy as np

class Events:
    def __init__(self):
        self._marks = []
        self._times = []

class BaselinesParameters:
    def __init__(self, unpacked: np.ndarray, num_of_marks: int, num_of_exps: int):
        self.unpacked = unpacked
        self.num_of_marks = num_of_marks
        self.num_of_exps = num_of_exps

    @property
    def packed(self):
        num_of_exps = self.num_of_exps
        num_of_marks = self.num_of_marks
        alphas = self.unpacked[:num_of_marks*num_of_exps].reshape(num_of_marks, num_of_exps)
        betas = self.unpacked[num_of_marks*num_of_exps:].reshape(num_of_marks, num_of_exps)
        return alphas, betas

class KernelsParameters:
    def __init__(self, unpacked: np.ndarray, num_of_marks: int):
        self.unpacked = unpacked
        self.num_of_marks = num_of_marks

    @property
    def packed(self):
        num_of_marks = self.num_of_marks
        alphas = self.unpacked[:num_of_marks*num_of_marks].reshape(num_of_marks, num_of_marks)
        betas = self.unpacked[num_of_marks*num_of_marks:].reshape(num_of_marks, num_of_marks)
        return alphas, betas


class Baselines:
    def __init__(self, params: BaselinesParameters):
        self.params = params
        alphas, betas = params.packed
        self.value = np.array([self._Baseline(alphas[i], betas[i]) for i in range(alphas.shape[0])])

    class _Baseline:
        def __init__(self, alphas, betas):
            self.alphas = alphas
            self.betas = betas

        def __call__(self, t):
            return np.sum(self.alphas * np.exp(-self.betas * t))

class Kernels:
    def __init__(self, params: KernelsParameters):
        self.params = params
        alphas, betas = params.packed
        self.value = np.array([[self._Kernel(alphas[i, j], betas[i, j]) for j in range(alphas.shape[1])] for i in range(alphas.shape[0])])

    def __getitem__(self, i):
        return self.value[i]

    class _Kernel:
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def __call__(self, t):
            return self.alpha * np.exp(-self.beta * t)

class Intensities:
    def __init__(self, baselines: Baselines, kernels: Kernels, events: Events):
        self.baselines = baselines
        self.kernels = kernels
        self.events = events

    def __call__(self, t):
        NotImplementedError()

class Compensators:
    def __init__(self, baselines: Baselines, kernels: Kernels, events: Events):
        self.baselines = baselines
        self.kernels = kernels
        self.events = events

    def __call__(self, t):
        NotImplementedError()


class Simulator:
    def set_parameters(self, b_params: BaselinesParameters, k_params: KernelsParameters):
        self.num_of_marks = b_params.num_of_marks
        self.b_params = b_params
        self.k_params = k_params
        self.baselines = Baselines(b_params)
        self.kernels = Kernels(k_params)

    def __call__(self, end_time):
        events = [np.empty(0) for _ in range(self.num_of_marks)]
        t = 0
        lambda_ast = np.sum(self.baselines(t))

        while t < end_time:
            # 平均 1/lambda_ast の指数分布に従う乱数Eを生成し、次のイベントの候補の発生時刻を t <- t + E とする
            E = np.random.exponential(1 / lambda_ast)
            t += E

            if t > end_time: break

            intensities = Intensities(self.baselines, self.kernels, events)(t)
            sum_intensities = np.sum(intensities)
            r = sum_intensities / lambda_ast

            # 一様乱数Uを生成し、rと比較する
            U = np.random.uniform()

            # U <= r ならば、イベントを採択する
            if U <= r:
                # どのノードでイベントが発生するかを決める
                k = np.random.choice(self.num_of_marks, p=intensities/sum_intensities)
                events[k] = np.append(events[k], t)
                lambda_ast = sum_intensities + np.sum([self._kernels[k, i](0) for i in range(self.num_of_marks)])
            else:
                lambda_ast = sum_intensities

        return events[k]

class LogLikelihood:
    def __init__(self, events: Events):
        self.events = events

    def __call__(self, b_params: BaselinesParameters, k_params: KernelsParameters):
        H_T = #TODO
        T = 1000 
        intensities = Intensities(Baselines(b_params), Kernels(k_params), self.events)
        compensators = Compensators(Baselines(b_params), Kernels(k_params), self.events)
        return -np.sum(compensators(T)) + np.sum(np.log([intensities(time)[mark] for (time, mark) in H_T]))

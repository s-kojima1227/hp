import numpy as np
from .thinning_method import ThinningMethod
from .output import Output
from ..function import Intensities

class Simulator:
    def __init__(self, t_interval=1):
        self.t_interval = t_interval

    def __call__(self, mu, kernel, T, params, kernel_type, type='thinning') -> Output:
        if type == 'thinning':
            events = ThinningMethod(mu, kernel)(T)
        else:
            raise NotImplementedError('現在シミュレーションは間引き法のみ対応しています')

        t = np.arange(0, T + self.t_interval, self.t_interval)
        intensities = Intensities(mu, kernel, events)

        return Output(
            events=events,
            t=t,
            intensity=intensities(t),
            params=params,
            kernel_type=kernel_type
        )


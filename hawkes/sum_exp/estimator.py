import numpy as np
from .function import LogLik, Intensities
from ..base import Estimator as Base, Loss
from .vo import Parameters as Params, ParametersFactory as PF

class Estimator(Base):
    def set_num_exps(self, num_exps):
        self._num_exps = num_exps

    @property
    def _loss_fn(self) -> Loss:
        loglik = LogLik(self._events)
        loglik.set_num_exps(self._num_exps)
        return loglik.to_loss()

    @property
    def _intensities_fn(self) -> Intensities:
        return Intensities(self._params_vo, self._events)

    @property
    def _params_vo(self) -> Params:
        return PF(self._dim, self._num_exps).build_from_unpacked(self._params)

    @property
    def _kernel_type(self):
        return 'sum_exp'

    @property
    def _default_minimization_config(self):
        dim = self._dim
        num_exps = self._num_exps

        dim_baselines = dim
        dim_adjacencies = dim * dim * num_exps
        dim_decays = num_exps
        sum_dim = dim_baselines + dim_adjacencies + dim_decays

        return {
            'method': 'scipy',
            'option': {
                'init_params': PF(dim, num_exps).build_from_unpacked(np.full(sum_dim, 0.1)),
                'bounds': [(1e-5, None)] * sum_dim,
            }
        }

    def set_minimization_config(self, method, option):
        if method == 'scipy':
            init_params = option.get('init_params')
            option['init_params'] = PF().build_from_dict(init_params)
        else:
            raise NotImplementedError('現在scipyのみ対応しています')

        super().set_minimization_config(method, option)

import numpy as np
from .function import LogLik, Intensities
from ..base import Estimator as Base, Loss
from .converter import BoundsConverter as BC
from .vo import Parameters as Params, ParametersFactory as PF

class Estimator(Base):
    @property
    def _loss_fn(self) -> Loss:
        return LogLik(self._events).to_loss()

    @property
    def _intensities_fn(self) -> Intensities:
        return Intensities(self._params_vo, self._events)

    @property
    def _params_vo(self) -> Params:
        return PF(self._dim).build_from_unpacked(self._params)

    @property
    def _kernel_type(self):
        return 'exp'

    @property
    def _default_minimization_config(self):
        dim = self._dim
        return {
            'method': 'scipy',
            'option': {
                'init_params': PF(dim).build_from_unpacked(np.full(dim + dim * dim + dim * dim, 0.1)),
                'bounds': [(1e-5, None)] * (dim + dim * dim + dim * dim),
            }
        }

    def set_minimization_config(self, method, option):
        if method == 'gradient' or method == 'scipy':
            init_params = option.get('init_params')
            option['init_params'] = PF().build_from_dict(init_params)
        if method == 'scipy' or method == 'random_search':
            bounds = option.get('bounds')
            bounds_mu = bounds.get('mu')
            bounds_a = bounds.get('a')
            bounds_b = bounds.get('b')
            option['bounds'] = \
                BC.pack(*BC.to_tensor(bounds_mu, bounds_a, bounds_b))
        if method == 'grid_search':
            grid = option.get('grid')
            grid_mu = grid.get('mu')
            grid_a = grid.get('a')
            grid_b = grid.get('b')
            option['grid'] = \
                BC.pack(*BC.to_tensor(grid_mu, grid_a, grid_b))

        super().set_minimization_config(method, option)

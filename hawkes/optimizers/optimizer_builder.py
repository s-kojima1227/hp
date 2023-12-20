from .gradient_optimizer import GradientOptimizer, GradientOptimizerSettings
from .random_search_optimizer import RandomSearchOptimizer, RandomSearchOptimizerSettings

class OptimizerBuilder:
    def __init__(self, optimizer_settings, dim, search_space, init_params, params_order):
        self._optimizer_settings = optimizer_settings
        self._dim = dim
        self._search_space = search_space
        self._init_params = init_params
        self._params_order = params_order

    def __call__(self):
        if isinstance(self._optimizer_settings, GradientOptimizerSettings):
            if self._dim != 1:
                raise ValueError('勾配法による推定は1次元のみ対応しています')
            if not self._optimizer_settings.is_init_param_set:
                self._optimizer_settings.init_param = self._init_params
            self._optimizer_settings.set_params_order(self._params_order)
            return GradientOptimizer(settings=self._optimizer_settings)

        if isinstance(self._optimizer_settings, RandomSearchOptimizerSettings):
            if self._optimizer_settings.search_space is None:
                self._optimizer_settings.search_space = self._search_space
            return RandomSearchOptimizer(settings=self._optimizer_settings)

        if self._optimizer_settings is None:
            self._optimizer_settings = RandomSearchOptimizerSettings()
            self._optimizer_settings.search_space = self._search_space
            return RandomSearchOptimizer(settings=self._optimizer_settings)

        raise ValueError('optimizer_settingsが不適切です')

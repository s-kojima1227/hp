from .gradient_method import GradientMethod
from .l_bfgs_b import L_MFGS_B
from .random_search import RandomSearch
from .grid_search import GridSearch

class MinimizerFactory:
    def __call__(self, config):
        method = config['method']
        option = config['option']
        if method == 'gradient':
            return GradientMethod(
                learning_rate=option.get('learning_rate', 0.0001),
                tol=option.get('tol', 0.01),
                max_iter=option.get('max_iter', 1000000),
                init_params=option.get('init_params'),
            )
        elif method == 'l_bfgs_b':
            return L_MFGS_B(
                init_params=option.get('init_params'),
                bounds=option.get('bounds'),
            )
        elif method == 'random_search':
            return RandomSearch(
                option.get('n_iter', 1000000),
                option.get('bounds'),
            )
        elif method == 'grid_search':
            return GridSearch(
                option.get('grid'),
            )
        else:
            raise ValueError('Unknown minimization method: {}'.format(self._minimization_config['method']))

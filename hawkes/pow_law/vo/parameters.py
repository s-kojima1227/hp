from ...base import Parameters as BaseP, ParametersFactory as BasePF
from typing import Tuple, Dict, Union
import numpy as np

class Parameters(BaseP):
    def __init__(self, unpacked: np.ndarray, dim: int):
        self._unpacked = unpacked
        self._dim = dim

    @property
    def unpacked(self) -> np.ndarray:
        return self._unpacked

    @property
    def dict(self) -> Dict:
        return {
            'baselines': self.baselines,
            'multipliers': self.multipliers,
            'exponents': self.exponents,
            'cutoffs': self.cutoffs,
        }

    @property
    def baselines(self) -> np.ndarray:
        return self._unpacked[:self._dim]

    @property
    def multipliers(self) -> np.ndarray:
        dim = self._dim
        return self._unpacked[dim:dim * (dim + 1)].reshape(dim, dim)

    @property
    def exponents(self) -> np.ndarray:
        dim = self._dim
        return self._unpacked[dim * (dim + 1):dim * (2 * dim + 1)].reshape(dim, dim)

    @property
    def cutoffs(self) -> np.ndarray:
        dim = self._dim
        return self._unpacked[dim * (2 * dim + 1):].reshape(dim, dim)

    @property
    def kernel_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.multipliers, self.exponents, self.cutoffs

class ParametersFactory(BasePF):
    def __init__(self, dim: Union[int, None]=None):
        self._dim = dim

    def build_from_packed(self, baselines, kernel_params) -> Parameters:
        multipliers, exponents, cutoffs = kernel_params
        if isinstance(baselines, (int, float)):
            baselines, multipliers, exponents, cutoffs = \
                (np.array([baselines]), np.array([[multipliers]]), np.array([[exponents]]), np.array([[cutoffs]]))
        else:
            baselines, multipliers, exponents, cutoffs = \
                (np.array(baselines), np.array(multipliers), np.array(exponents), np.array(cutoffs))

        dim = baselines.shape[0]
        unpacked = np.concatenate([baselines, multipliers.flatten(), exponents.flatten(), cutoffs.flatten()])

        return Parameters(unpacked, dim)

    def build_from_unpacked(self, params: np.ndarray) -> Parameters:
        if self._dim is None:
            raise ValueError('dim is not set')
        return Parameters(params, self._dim)

    def build_from_dict(self, params: Dict) -> Parameters:
        baselines = params.get('baselines')
        multipliers = params.get('multipliers')
        exponents = params.get('exponents')
        cutoffs = params.get('cutoffs')
        return self.build_from_packed(baselines, (multipliers, exponents, cutoffs))
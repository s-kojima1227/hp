from abc import ABCMeta, abstractmethod

class BaseKernel(metaclass=ABCMeta):
    """カーネル関数の基底クラス"""

    @abstractmethod
    def __call__(self, t_values):
        """カーネル関数の値を返す"""
        pass

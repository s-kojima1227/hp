from .simulator import Simulator
from .estimator import Estimator

class Model:
    @staticmethod
    def build_estimator(optimizer='scipy'):
        return Estimator()

    @staticmethod
    def build_simulator():
        return Simulator()

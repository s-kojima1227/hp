import numpy as np

class FunctionIO:
    def __init__(self, input, output):
        self._input = input
        self._output = output

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

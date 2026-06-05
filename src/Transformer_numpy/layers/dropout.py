import numpy as np


class Dropout:
    def __init__(self, rate=0.0):
        self.rate = rate
        self.training = True
        self._mask = None

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * self._mask

    def backward(self, grad):
        if not self.training or self.rate == 0.0:
            return grad
        return grad * self._mask

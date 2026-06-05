"""
feed_forward.py
Feed Forward Layer for NumPy Transformer
Provides: FeedForward
"""

import numpy as np
from typing import Optional
from src.Transformer_numpy.utils import gelu, gelu_derivative
from src.Transformer_numpy.layers.linear import Linear
from src.Transformer_numpy.layers.dropout import Dropout
from src.Transformer_numpy import config

class FeedForward:
    """
    Position-wise FFN: Linear -> GELU -> Dropout -> Linear
    """

    def __init__(self, embed_dim: int, dropout_rate: float = config.DROPOUT_RATE):
        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.fc2 = Linear(4 * embed_dim, embed_dim)
        self.dropout = Dropout(dropout_rate)

        self._pre_act: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1.forward(x)
        self._pre_act = x
        x = gelu(x)
        x = self.dropout.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.fc2.backward(grad)
        grad = self.dropout.backward(grad)
        grad = grad * gelu_derivative(self._pre_act)
        return self.fc1.backward(grad)

    def get_params(self):
        return {'fc1': self.fc1.get_params(), 'fc2': self.fc2.get_params()}
    
    def get_grads(self):
        return {'fc1': self.fc1.get_grads(), 'fc2': self.fc2.get_grads()}
    
    def set_params(self, params):
        self.fc1.set_params(params['fc1'])
        self.fc2.set_params(params['fc2'])
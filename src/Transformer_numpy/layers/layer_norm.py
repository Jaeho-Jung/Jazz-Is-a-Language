"""
layer_norm.py
Layer Normalization Layer
Provides: LayerNorm
"""

from typing import Optional
import numpy as np

class LayerNorm:
    """
    Layer normalization over the last axis.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma: np.ndarray = np.ones(normalized_shape)
        self.beta: np.ndarray = np.zeros(normalized_shape)
        self.grad_gamma: np.ndarray = np.zeros_like(self.gamma)
        self.grad_beta: np.ndarray = np.zeros_like(self.beta)

        self._x_norm: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer normalization layer.
        
        Mathematically: y = gamma * (x - mean) / std + beta

        Args:
            x: (..., normalized_shape)
        Returns:
            y: (..., normalized_shape)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self._std = np.sqrt(var + self.eps)
        self._x_norm = (x - mean) / self._std
        return self.gamma * self._x_norm + self.beta
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer normalization layer.
        
        Args:
            grad_output: Gradient of the output, shape (..., normalized_shape)
        Returns:
            grad_input: Gradient of the input, shape (..., normalized_shape)
        """
        x_norm = self._x_norm
        reduce_axes = tuple(range(grad_output.ndim - 1))

        self.grad_gamma = np.sum(grad_output * x_norm, axis=reduce_axes)
        self.grad_beta = np.sum(grad_output, axis=reduce_axes)
        
        dx_norm = grad_output * self.gamma

        # Standard LN gradient formula
        dx = (1.0 / self._std) * (dx_norm - np.mean(dx_norm, axis=-1, keepdims=True) - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True))

        return dx
    
    def get_params(self):
        """Return parameters for optimizer."""
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def get_grads(self):
        """Return gradients for optimizer."""
        return {'gamma': self.grad_gamma, 'beta': self.grad_beta}
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.gamma = params['gamma']
        self.beta = params['beta']
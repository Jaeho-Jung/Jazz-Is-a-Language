"""
linear.py
Linear Layer
Provides: Linear
"""

import numpy as np
from typing import Optional

class Linear:
    """
    Fully connected layer for a neural network.
    
    Mathematically: output = x @ W^T + b
    where x ∈ ℝ^(batch_size × input_dim), W ∈ ℝ^(input_dim × output_dim), b ∈ ℝ^(output_dim)
    """
    
    def __init__(self, input_features: int, output_features: int):
        self.input_features = input_features
        self.output_features = output_features
        
        # Initialize weights and biases
        self.W = np.random.randn(output_features, input_features) * np.sqrt(2.0 / input_features)
        self.b = np.zeros(output_features)
        
        self.grad_W: np.ndarray = None
        self.grad_b: np.ndarray = None
        
        self._x: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the linear layer.
        
        Args:
            x: Input, shape (batch_size, input_features)
        
        Returns:
            z: Output, shape (batch_size, output_features)
        """
        self._x = x
        z = x @ self.W.T + self.b
        return z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the linear layer.
        
        Args:
            grad_output: Gradient from next layer, shape (batch_size, output_features)
        
        Returns:
            grad_x: Gradient w.r.t. input, shape (batch_size, input_features)
        """
        grad_x = grad_output @ self.W
        self.grad_W = grad_output.T @ self._x
        self.grad_b = np.sum(grad_output, axis=0)
        
        return grad_x

    def get_params(self):
        """Return parameters for optimizer."""
        return {'W': self.W, 'b': self.b}
    
    def get_grads(self):
        """Return gradients for optimizer."""
        return {'W': self.grad_W, 'b': self.grad_b}
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.W = params['W']
        self.b = params['b']
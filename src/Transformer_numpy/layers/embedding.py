"""
embedding.py
Embedding Layer for NumPy Transformer
Provides: Embedding
"""

import numpy as np
from typing import Dict, Optional

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # W ∈ ℝ^(num_embeddings × embedding_dim)
        self.W: np.ndarray = np.random.randn(num_embeddings, embedding_dim) * np.sqrt(2.0 / num_embeddings)

        self.grad_W: np.ndarray = None

        self._indices: Optional[np.ndarray] = None

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        return self.forward(indices)


    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Forward pass through the embedding layer.
        
        Args:
            indices: (batch_size, seq_len) or (batch_size, seq_len, 1)
        Returns:
            embedded: (batch_size, seq_len, embedding_dim)
        """
        # Validate input
        assert np.all(indices >= 0) and np.all(indices < self.num_embeddings), \
            f"Indices must be in [0, {self.num_embeddings}), got min={indices.min()}, max={indices.max()}"


        # Store input for backward pass
        self._indices = indices
        
        # Get embeddings
        embeddings = self.W[indices]
        
        return embeddings

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass through the embedding layer.
        
        Args:
            grad_output: Gradient of the output, shape (batch_size, seq_len, embedding_dim)
        Returns:
            grad_indices: Gradient of the input indices, shape (batch_size, seq_len)
        """
        # Get indices from cache
        indices = self._indices
        
        # Calculate gradient of W
        self.grad_W = np.zeros_like(self.W)

        flat_indices = indices.flatten()
        flat_grad_output = grad_output.reshape(-1, self.embedding_dim)

        np.add.at(self.grad_W, flat_indices, flat_grad_output)
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Return parameters for optimizer."""
        return {'W': self.W}
    
    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return gradients for optimizer."""
        return {'W': self.grad_W}
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.W = params['W']
"""
Embedding Layer for NumPy LSTM
"""

import numpy as np


class Embedding:
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    
    Mathematically: output = W[indices]
    where W ∈ ℝ^(num_embeddings × embedding_dim)
    
    Args:
        num_embeddings (int): Size of the dictionary of embeddings (vocabulary size)
        embedding_dim (int): The size of each embedding vector
    
    Example:
        >>> embed = Embedding(num_embeddings=100, embedding_dim=16)
        >>> indices = np.array([5, 23, 7])  # Batch of 3
        >>> vectors = embed.forward(indices)  # Shape: (3, 16)
    """
    
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialize embedding matrix with small random values
        # Using Xavier/Glorot initialization scaled for embeddings
        # W ∈ ℝ^(num_embeddings × embedding_dim)
        self.W = np.random.randn(num_embeddings, embedding_dim) * np.sqrt(2.0 / num_embeddings)

        # Gradient accumulator (populated during backward pass)
        self.grad_W = None

        # Cache for backward pass
        self.cache = None

    def forward(self, indices):
        """
        Forward pass: Lookup embeddings for given indices.
        
        Args:
            indices: np.ndarray of integers, shape (batch,) or (batch, seq_len)
                     Each value must be in [0, num_embeddings)
        
        Returns:
            embeddings: np.ndarray of shape (batch, embedding_dim) 
                        or (batch, seq_len, embedding_dim)
        
        Example:
            >>> embed = Embedding(num_embeddings=100, embedding_dim=16)
            >>> indices = np.array([5, 23, 7])  # Batch of 3
            >>> vectors = embed.forward(indices)  # Shape: (3, 16)
        """
        # Validate input
        assert np.all(indices >= 0) and np.all(indices < self.num_embeddings), \
            f"Indices must be in [0, {self.num_embeddings}), got min={indices.min()}, max={indices.max()}"

        # Cache indices for backward pass
        self.cache = indices

        # Lookup: Simple array indexing
        embeddings = self.W[indices]
        
        return embeddings
        
    def backward(self, grad_output):
        """
        Backward pass: Accumulate gradients for embedding weights.
        
        Mathematical formulation:
            ∂L/∂W[i] = Σ_{positions where index i was used} ∂L/∂output[position]
        
        Args:
            grad_output: Gradient from next layer, same shape as forward() output
                         Shape: (batch, embedding_dim) or (batch, seq_len, embedding_dim)
        
        Returns:
            None (no gradient w.r.t. discrete indices)
        
        Side effects:
            Sets self.grad_W to accumulated gradients
        
        Example:
            >>> indices = np.array([0, 2, 0])  # Index 0 used twice
            >>> grad_output = np.array([[1, 2], [3, 4], [5, 6]])
            >>> backward(grad_output)
            >>> # grad_W[0] = [1,2] + [5,6] = [6, 8]  (accumulated)
            >>> # grad_W[2] = [3, 4]
        """
        indices = self.cache

        # Initialize gradient matrix with zeros
        self.grad_W = np.zeros_like(self.W)

        # Flatten indices and gradients to handle any input shape
        flat_indices = indices.flatten()
        flat_grad_output = grad_output.reshape(-1, self.embedding_dim)

        np.add.at(self.grad_W, flat_indices, flat_grad_output)
        
    def get_params(self):
        """Return parameters for optimizer."""
        return {'W': self.W}
    
    def get_grads(self):
        """Return gradients for optimizer."""
        return {'W': self.grad_W}
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.W = params['W']
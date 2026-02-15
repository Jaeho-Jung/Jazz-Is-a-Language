"""
Vanilla RNN Cell Implementation

Mathematical formulation:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
"""

import numpy as np
from src.RNN_numpy.utils import tanh, tanh_derivative

class RNNCell:
    """
    Vanilla RNN cell for processing a single timestep.
    
    The cell maintains a hidden state that combines:
    - Current input (transformed by W_xh)
    - Previous hidden state (transformed by W_hh)
    
    Args:
        input_size (int): Dimension of input vector x_t
        hidden_size (int): Dimension of hidden state h_t
    
    Example:
        >>> rnn = RNNCell(input_size=10, hidden_size=20)
        >>> x_t = np.random.randn(batch_size, 10)
        >>> h_prev = np.zeros((batch_size, 20))
        >>> h_t = rnn.forward(x_t, h_prev)
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights using Xavier/Glorot initialization
        # W_xh: Transforms input to hidden space
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)

        # W_hh: Recurrent weights (hidden-to-hidden)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)

        # Bias
        self.b_h = np.zeros(hidden_size)

        # Gradients
        self.grad_W_xh = None
        self.grad_W_hh = None
        self.grad_b_h = None

        # Cache for backward pass
        self.cache = None

    def forward(self, x_t, h_prev):
        """
        Forward pass for a single timestep 
        Computation:
            1. z_t = x_t @ W_xh.T + h_prev @ W_hh.T + b_h
            2. h_t = tanh(z_t)
        
        Args:
            x_t: Input at time t, shape (batch, input_size)
            h_prev: Previous hidden state, shape (batch, hidden_size)
        
        Returns:
            h_t: New hidden state, shape (batch, hidden_size)
        """
        # Linear transformation
        # Note: Using @ W.T for (batch, features) @ (features, hidden)^T = (batch, hidden)
        z_t = x_t  @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h

        # Activation: tanh squashes to (-1, 1)
        h_t = tanh(z_t)

        self.cache = {
            'x_t': x_t, 
            'h_prev': h_prev,
            'z_t': z_t,
            'h_t': h_t
        }

        return h_t

    def backward(self, grad_h_t):
        """
        Backward pass for a single timestep (BPTT).

        Given ∂L/∂h_t, compute:
          - ∂L/∂W_xh, ∂L/∂W_hh, ∂L/∂b_h (parameter gradients)
          - ∂L/∂x_t (gradient w.r.t. input)
          - ∂L/∂h_{t-1} (gradient to pass to previous timestep)
        
        Derivation:
            h_t = tanh(z_t)
            z_t = x_t @ W_xh.T + h_prev @ W_hh.T + b_h
            
            ∂L/∂z_t = ∂L/∂h_t ⊙ (1 - tanh²(z_t))
                    = ∂L/∂h_t ⊙ (1 - h_t²)
            
            ∂L/∂W_xh = ∂L/∂z_t^T @ x_t
            ∂L/∂W_hh = ∂L/∂z_t^T @ h_prev
            ∂L/∂b_h = sum(∂L/∂z_t, axis=0)
            
            ∂L/∂x_t = ∂L/∂z_t @ W_xh
            ∂L/∂h_{t-1} = ∂L/∂z_t @ W_hh
        
        Args:
            grad_h_t: Gradient from next timestep, shape (batch, hidden_size)
        
        Returns:
            grad_x_t: Gradient w.r.t. input, shape (batch, input_size)
            grad_h_prev: Gradient w.r.t. previous hidden state, shape (batch, hidden_size)
        """
        # Retrieve cached values
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        h_t = self.cache['h_t']

        # Gradient through tanh activation
        grad_z_t = grad_h_t * tanh_derivative(h_t)

        # Weight gradients (matrix multiplication)
        # For batched input, we sum gradients across the batch
        self.grad_W_xh = grad_z_t.T @ x_t
        self.grad_W_hh = grad_z_t.T @ h_prev
        self.grad_b_h = np.sum(grad_z_t, axis=0)

        # Gradients to pass backward
        grad_x_t = grad_z_t @ self.W_xh
        grad_h_prev = grad_z_t @ self.W_hh

        return grad_x_t, grad_h_prev

    def get_params(self):
        """Return parameters for optimizer."""
        return {
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'b_h': self.b_h
        }
    
    def get_grads(self):
        """Return gradients for optimizer."""
        return {
            'W_xh': self.grad_W_xh,
            'W_hh': self.grad_W_hh,
            'b_h': self.grad_b_h
        }
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.W_xh = params['W_xh']
        self.W_hh = params['W_hh']
        self.b_h = params['b_h']

class RNN:
    """
    Multi-timestep RNN that processes entire sequences.

    This wraps RNNCel and handles the temporal loop.
    """

    def __init__(self, input_size, hidden_size):
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq, h_0=None):
        """
        Process a full sequence.

        Args:
            x_seq: Input sequence, shape (batch, seq_len, input_size)
            h_0: Initial hidden state, shape (batch, hidden_size)
                 If None, initialized to zeros

        Returns:
            h_seq: Hidden states for all timesteps, shape (batch, seq_len, hidden_size)
            h_final: Final hidden state, shape (batch, hidden_size)
        """
        batch_size, seq_len, input_size = x_seq.shape

        # Initialize hidden state if not provided
        if h_0 is None:
            h_t = np.zeros((batch_size, self.hidden_size))
        else:
            h_t = h_0

        # Process sequence
        h_seq = []
        for t in range(seq_len):
            h_t = self.cell.forward(x_seq[:, t, :], h_t)
            h_seq.append(h_t)

        # Stack hidden states: list of (batch, hidden) -> (batch, seq_len, hidden)
        h_seq = np.stack(h_seq, axis=1)

        return h_seq, h_t

    def backward(self, grad_h_seq):
        """
        Backward pass through entire sequence.

        Args:
            grad_h_seq: Gradients for all timesteps, shape (batch, seq_len, hidden_size)
        
        Returns:
            grad_x_seq: Gradients w.r.t. input sequence, shape (batch, seq_len, input_size)
        """
        seq_len = grad_h_seq.shape[1]
        grad_x_seq = []

        # Initialize gradient flowing backward through time
        grad_h_t = np.zeros_like(grad_h_seq[:, 0, :])
        # Process sequence in reverse order
        for t in reversed(range(seq_len)):
            # Add gradient from current timestep
            grad_h_t += grad_h_seq[:, t, :]

            # Backpropagate through cell
            grad_x_t, grad_h_t = self.cell.backward(grad_h_t)
            grad_x_seq.insert(0, grad_x_t)

        # Stack gradients
        grad_x_seq = np.stack(grad_x_seq, axis=1)

        return grad_x_seq
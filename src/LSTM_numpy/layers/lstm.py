"""
Vanilla RNN Cell Implementation

Mathematical formulation:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
"""

import numpy as np
from src.LSTM_numpy.utils import tanh, sigmoid, tanh_derivative

class LSTMCell:
    """
    Vanilla LSTM cell for processing a single timestep.
    
    Args:
        input_size (int): Dimension of input vector x_t
        hidden_size (int): Dimension of hidden state h_t
    
    Example:
        >>> lstm = LSTMCell(input_size=10, hidden_size=20)
        >>> x_t = np.random.randn(batch_size, 10)
        >>> h_prev = np.zeros((batch_size, 20))
        >>> h_t = lstm.forward(x_t, h_prev)
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights using Xavier/Glorot initialization
        # U: input weights
        self.U_ifo = np.random.randn(3 * hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.U_cell = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)

        # W: recurrent weights (hidden-to-hidden)
        self.W_ifo = np.random.randn(3 * hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_cell = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)

        # Bias
        self.b_ifo = np.zeros(3 * hidden_size)
        self.b_cell = np.zeros(hidden_size)

        # Gradients
        self.grad_U_ifo = None
        self.grad_U_cell = None
        self.grad_W_ifo = None
        self.grad_W_cell = None
        self.grad_b_ifo = None
        self.grad_b_cell = None

        # Cache for backward pass
        self.cache = None

    def forward(self, x_t, h_prev, c_prev):
        """
        Forward pass for a single timestep 
        Computation:
            1. 
        
        Args:
            x_t: Input at time t, shape (batch, input_size)
            h_prev: Previous hidden state, shape (batch, hidden_size)
            c_prev: Previous cell state, shape (batch, hidden_size)
        
        Returns:
            h_t: New hidden state, shape (batch, hidden_size)
            c_t: New cell state, shape (batch, hidden_size)
        """
        # Forget, Input, Output gates
        z_ifo = x_t @ self.U_ifo.T + h_prev @ self.W_ifo.T + self.b_ifo
        
        # Candidate cell state
        z_cell = x_t @ self.U_cell.T + h_prev @ self.W_cell.T + self.b_cell

        # Activation
        i_t, f_t, o_t = np.split(sigmoid(z_ifo), 3, axis=1)
        g_t = tanh(z_cell)
        
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)

        self.cache = {
            'x_t': x_t, 
            'h_prev': h_prev,
            'c_prev': c_prev,
            'f_t': f_t,
            'i_t': i_t,
            'o_t': o_t,
            'g_t': g_t,
            'c_t': c_t,
            'h_t': h_t
        }

        return h_t, c_t

    def backward(self, grad_h_t, grad_c_t):
        """
        Backward pass for a single timestep (BPTT).

        Given ∂L/∂h_t, compute:
        
        Derivation:
        
        Args:
            grad_h_t: Gradient from next timestep, shape (batch, hidden_size)
            grad_c_t: Gradient from next timestep, shape (batch, hidden_size)
        
        Returns:
            grad_x_t: Gradient w.r.t. input, shape (batch, input_size)
            grad_h_prev: Gradient w.r.t. previous hidden state, shape (batch, hidden_size)
        """
        # Retrieve cached values
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        c_prev = self.cache['c_prev']
        f_t = self.cache['f_t']
        i_t = self.cache['i_t']
        o_t = self.cache['o_t']
        g_t = self.cache['g_t']
        c_t = self.cache['c_t']
        h_t = self.cache['h_t']

        # Gradient through output gate -> cell state
        # h_t = o_t * tanh(c_t)
        # d h_t / d c_t = o_t * (1 - tanh(c_t)^2)
        tanh_c_t = tanh(c_t)
        grad_c_from_h = grad_h_t * o_t * (1 - tanh_c_t ** 2)
        
        # Total gradient on c_t (from h_t and from next timestep)
        grad_c_total = grad_c_from_h + grad_c_t
        
        # Gradient through cell state to c_prev
        # c_t = f_t * c_prev + i_t * g_t
        grad_c_prev = grad_c_total * f_t

        # Gradient through gates (with sigmoid derivative: σ'(z) = σ(z)(1-σ(z)))
        grad_i_t = grad_c_total * g_t * i_t * (1 - i_t)
        grad_f_t = grad_c_total * c_prev * f_t * (1 - f_t)
        grad_o_t = grad_h_t * tanh_c_t * o_t * (1 - o_t)
        grad_g_t = grad_c_total * i_t * (1 - g_t ** 2)

        # Weight gradients (matrix multiplication)
        # For batched input, we sum gradients across the batch
        grad_z_t = np.hstack([grad_i_t, grad_f_t, grad_o_t])
        self.grad_W_ifo = grad_z_t.T @ h_prev
        self.grad_W_cell = grad_g_t.T @ h_prev
        self.grad_b_ifo = np.sum(grad_z_t, axis=0)
        self.grad_b_cell = np.sum(grad_g_t, axis=0)

        self.grad_U_ifo = grad_z_t.T @ x_t
        self.grad_U_cell = grad_g_t.T @ x_t
        self.grad_b_ifo = np.sum(grad_z_t, axis=0)
        self.grad_b_cell = np.sum(grad_g_t, axis=0)

        grad_h_prev = grad_z_t @ self.W_ifo + grad_g_t @ self.W_cell

        # Gradients to pass backward
        grad_x_t = grad_z_t @ self.U_ifo + grad_g_t @ self.U_cell

        return grad_x_t, grad_h_prev, grad_c_prev

    def get_params(self):
        """Return parameters for optimizer."""
        return {
            'U_ifo': self.U_ifo,
            'U_cell': self.U_cell,
            'W_ifo': self.W_ifo,
            'W_cell': self.W_cell,
            'b_ifo': self.b_ifo,
            'b_cell': self.b_cell
        }
    
    def get_grads(self):
        """Return gradients for optimizer."""
        return {
            'U_ifo': self.grad_U_ifo,
            'U_cell': self.grad_U_cell,
            'W_ifo': self.grad_W_ifo,
            'W_cell': self.grad_W_cell,
            'b_ifo': self.grad_b_ifo,
            'b_cell': self.grad_b_cell
        }
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.U_ifo = params['U_ifo']
        self.U_cell = params['U_cell']
        self.W_ifo = params['W_ifo']
        self.W_cell = params['W_cell']
        self.b_ifo = params['b_ifo']
        self.b_cell = params['b_cell']

class LSTM:
    """
    Multi-timestep LSTM that processes entire sequences.

    This wraps LSTMCell and handles the temporal loop.
    """

    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq, h_0=None, c_0=None):
        """
        Process a full sequence.

        Args:
            x_seq: Input sequence, shape (batch, seq_len, input_size)
            h_0: Initial hidden state, shape (batch, hidden_size)
                 If None, initialized to zeros
            c_0: Initial cell state, shape (batch, hidden_size)
                 If None, initialized to zeros

        Returns:
            h_seq: Hidden states for all timesteps, shape (batch, seq_len, hidden_size)
            h_final: Final hidden state, shape (batch, hidden_size)
            c_final: Final cell state, shape (batch, hidden_size)
        """
        batch_size, seq_len, input_size = x_seq.shape

        # Initialize hidden state if not provided
        if h_0 is None:
            h_t = np.zeros((batch_size, self.hidden_size))
        else:
            h_t = h_0
            
        # Initialize cell state if not provided
        if c_0 is None:
            c_t = np.zeros((batch_size, self.hidden_size))
        else:
            c_t = c_0

        # Store cache for backward pass
        self.cache_list = []
        
        # Process sequence
        h_seq = []
        for t in range(seq_len):
            h_t, c_t = self.cell.forward(x_seq[:, t, :], h_t, c_t)
            h_seq.append(h_t)
            self.cache_list.append(self.cell.cache.copy())

        # Stack hidden states: list of (batch, hidden) -> (batch, seq_len, hidden)
        h_seq = np.stack(h_seq, axis=1)

        return h_seq, h_t, c_t

    def backward(self, grad_h_seq, grad_h_final=None, grad_c_final=None):
        """
        Backward pass through entire sequence.

        Args:
            grad_h_seq: Gradients for all timesteps, shape (batch, seq_len, hidden_size)
            grad_h_final: Gradient from final hidden state (optional)
            grad_c_final: Gradient from final cell state (optional)
        
        Returns:
            grad_x_seq: Gradients w.r.t. input sequence, shape (batch, seq_len, input_size)
        """
        seq_len = grad_h_seq.shape[1]
        batch_size = grad_h_seq.shape[0]
        grad_x_seq = []
        
        # Initialize accumulated gradients for parameters
        self.grad_U_ifo = np.zeros_like(self.cell.U_ifo)
        self.grad_U_cell = np.zeros_like(self.cell.U_cell)
        self.grad_W_ifo = np.zeros_like(self.cell.W_ifo)
        self.grad_W_cell = np.zeros_like(self.cell.W_cell)
        self.grad_b_ifo = np.zeros_like(self.cell.b_ifo)
        self.grad_b_cell = np.zeros_like(self.cell.b_cell)

        # Initialize gradient flowing backward through time
        grad_h_t = np.zeros((batch_size, self.hidden_size))
        grad_c_t = np.zeros((batch_size, self.hidden_size))
        
        if grad_h_final is not None:
            grad_h_t = grad_h_final.copy()
        if grad_c_final is not None:
            grad_c_t = grad_c_final.copy()
            
        # Process sequence in reverse order
        for t in reversed(range(seq_len)):
            # Restore cache for this timestep
            self.cell.cache = self.cache_list[t]
            
            # Add gradient from current timestep output
            grad_h_t = grad_h_t + grad_h_seq[:, t, :]

            # Backpropagate through cell
            grad_x_t, grad_h_t, grad_c_t = self.cell.backward(grad_h_t, grad_c_t)
            grad_x_seq.insert(0, grad_x_t)
            
            # Accumulate parameter gradients
            self.grad_U_ifo += self.cell.grad_U_ifo
            self.grad_U_cell += self.cell.grad_U_cell
            self.grad_W_ifo += self.cell.grad_W_ifo
            self.grad_W_cell += self.cell.grad_W_cell
            self.grad_b_ifo += self.cell.grad_b_ifo
            self.grad_b_cell += self.cell.grad_b_cell

        # Stack gradients
        grad_x_seq = np.stack(grad_x_seq, axis=1)

        return grad_x_seq
    
    def get_params(self):
        """Return parameters for optimizer."""
        return self.cell.get_params()
    
    def get_grads(self):
        """Return accumulated gradients for optimizer."""
        return {
            'U_ifo': self.grad_U_ifo,
            'U_cell': self.grad_U_cell,
            'W_ifo': self.grad_W_ifo,
            'W_cell': self.grad_W_cell, 
            'b_ifo': self.grad_b_ifo,
            'b_cell': self.grad_b_cell
        }
    
    def set_params(self, params):
        """Update parameters."""
        self.cell.set_params(params)
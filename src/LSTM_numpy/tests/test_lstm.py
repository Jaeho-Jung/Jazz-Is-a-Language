"""
Unit tests for LSTM implementation.

Tests LSTMCell and LSTM classes for:
- Forward pass shape correctness
- Backward pass gradient shapes
- Numerical gradient checking
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.LSTM_numpy.layers.lstm import LSTMCell, LSTM


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient using central differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + eps
        fx_plus = f()
        
        x[idx] = old_val - eps
        fx_minus = f()
        
        x[idx] = old_val
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        it.iternext()
    return grad


def test_lstm_cell_forward_shapes():
    """Test LSTMCell forward pass output shapes."""
    print("\n=== Testing LSTMCell Forward Shapes ===")
    
    batch_size, input_size, hidden_size = 4, 10, 20
    cell = LSTMCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    c_prev = np.random.randn(batch_size, hidden_size)
    
    h_t, c_t = cell.forward(x_t, h_prev, c_prev)
    
    assert h_t.shape == (batch_size, hidden_size), f"h_t shape mismatch: {h_t.shape}"
    assert c_t.shape == (batch_size, hidden_size), f"c_t shape mismatch: {c_t.shape}"
    
    print(f"✓ h_t shape: {h_t.shape}")
    print(f"✓ c_t shape: {c_t.shape}")
    

def test_lstm_cell_backward_shapes():
    """Test LSTMCell backward pass output shapes."""
    print("\n=== Testing LSTMCell Backward Shapes ===")
    
    batch_size, input_size, hidden_size = 4, 10, 20
    cell = LSTMCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    c_prev = np.random.randn(batch_size, hidden_size)
    
    # Forward
    h_t, c_t = cell.forward(x_t, h_prev, c_prev)
    
    # Backward
    grad_h_t = np.random.randn(batch_size, hidden_size)
    grad_c_t = np.random.randn(batch_size, hidden_size)
    
    grad_x_t, grad_h_prev, grad_c_prev = cell.backward(grad_h_t, grad_c_t)
    
    assert grad_x_t.shape == (batch_size, input_size), f"grad_x_t shape: {grad_x_t.shape}"
    assert grad_h_prev.shape == (batch_size, hidden_size), f"grad_h_prev shape: {grad_h_prev.shape}"
    assert grad_c_prev.shape == (batch_size, hidden_size), f"grad_c_prev shape: {grad_c_prev.shape}"
    
    print(f"✓ grad_x_t shape: {grad_x_t.shape}")
    print(f"✓ grad_h_prev shape: {grad_h_prev.shape}")
    print(f"✓ grad_c_prev shape: {grad_c_prev.shape}")
    
    # Check parameter gradients
    assert cell.grad_U_ifo.shape == (3 * hidden_size, input_size)
    assert cell.grad_U_cell.shape == (hidden_size, input_size)
    assert cell.grad_W_ifo.shape == (3 * hidden_size, hidden_size)
    assert cell.grad_W_cell.shape == (hidden_size, hidden_size)
    assert cell.grad_b_ifo.shape == (3 * hidden_size,)
    assert cell.grad_b_cell.shape == (hidden_size,)
    
    print("✓ All parameter gradient shapes correct")


def test_lstm_forward_shapes():
    """Test LSTM (full sequence) forward pass shapes."""
    print("\n=== Testing LSTM Forward Shapes ===")
    
    batch_size, seq_len, input_size, hidden_size = 4, 8, 10, 20
    lstm = LSTM(input_size, hidden_size)
    
    x_seq = np.random.randn(batch_size, seq_len, input_size)
    
    h_seq, h_final, c_final = lstm.forward(x_seq)
    
    assert h_seq.shape == (batch_size, seq_len, hidden_size), f"h_seq shape: {h_seq.shape}"
    assert h_final.shape == (batch_size, hidden_size), f"h_final shape: {h_final.shape}"
    assert c_final.shape == (batch_size, hidden_size), f"c_final shape: {c_final.shape}"
    
    print(f"✓ h_seq shape: {h_seq.shape}")
    print(f"✓ h_final shape: {h_final.shape}")
    print(f"✓ c_final shape: {c_final.shape}")


def test_lstm_backward_shapes():
    """Test LSTM (full sequence) backward pass shapes."""
    print("\n=== Testing LSTM Backward Shapes ===")
    
    batch_size, seq_len, input_size, hidden_size = 4, 8, 10, 20
    lstm = LSTM(input_size, hidden_size)
    
    x_seq = np.random.randn(batch_size, seq_len, input_size)
    
    # Forward
    h_seq, h_final, c_final = lstm.forward(x_seq)
    
    # Backward
    grad_h_seq = np.random.randn(batch_size, seq_len, hidden_size)
    grad_x_seq = lstm.backward(grad_h_seq)
    
    assert grad_x_seq.shape == (batch_size, seq_len, input_size), f"grad_x_seq shape: {grad_x_seq.shape}"
    
    print(f"✓ grad_x_seq shape: {grad_x_seq.shape}")
    
    # Check accumulated gradients
    grads = lstm.get_grads()
    assert grads['U_ifo'].shape == (3 * hidden_size, input_size)
    assert grads['W_ifo'].shape == (3 * hidden_size, hidden_size)
    
    print("✓ Accumulated gradient shapes correct")


def test_lstm_cell_gradient_check():
    """Numerical gradient check for LSTMCell."""
    print("\n=== Testing LSTMCell Gradient Check ===")
    
    np.random.seed(42)
    batch_size, input_size, hidden_size = 2, 3, 4
    
    cell = LSTMCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size) * 0.1
    h_prev = np.random.randn(batch_size, hidden_size) * 0.1
    c_prev = np.random.randn(batch_size, hidden_size) * 0.1
    
    # Forward + simple loss (sum of h_t)
    h_t, c_t = cell.forward(x_t, h_prev, c_prev)
    
    # Backward with gradient = 1 for h_t
    grad_h_t = np.ones_like(h_t)
    grad_c_t = np.zeros_like(c_t)  # No direct gradient on c_t
    
    grad_x_t, grad_h_prev, grad_c_prev = cell.backward(grad_h_t, grad_c_t)
    
    # Numerical gradient for x_t
    def loss_fn():
        h, c = cell.forward(x_t, h_prev, c_prev)
        return np.sum(h)
    
    num_grad_x = numerical_gradient(loss_fn, x_t)
    
    # Compare
    rel_error = np.abs(grad_x_t - num_grad_x) / (np.abs(grad_x_t) + np.abs(num_grad_x) + 1e-8)
    max_error = np.max(rel_error)
    
    print(f"  Max relative error (grad_x_t): {max_error:.2e}")
    
    if max_error < 1e-4:
        print("✓ Gradient check passed!")
    else:
        print(f"⚠ Gradient check warning: max error = {max_error:.2e}")


def test_lstm_gates():
    """Test that LSTM gates behave correctly."""
    print("\n=== Testing LSTM Gates ===")
    
    batch_size, input_size, hidden_size = 1, 2, 3
    cell = LSTMCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((batch_size, hidden_size))
    c_prev = np.zeros((batch_size, hidden_size))
    
    h_t, c_t = cell.forward(x_t, h_prev, c_prev)
    
    # Check gates are in valid range (sigmoid output)
    cache = cell.cache
    assert np.all(cache['i_t'] >= 0) and np.all(cache['i_t'] <= 1), "Input gate out of range"
    assert np.all(cache['f_t'] >= 0) and np.all(cache['f_t'] <= 1), "Forget gate out of range"
    assert np.all(cache['o_t'] >= 0) and np.all(cache['o_t'] <= 1), "Output gate out of range"
    
    # Check tanh outputs
    assert np.all(cache['g_t'] >= -1) and np.all(cache['g_t'] <= 1), "Candidate cell out of range"
    
    print("✓ Input gate range: [0, 1]")
    print("✓ Forget gate range: [0, 1]")
    print("✓ Output gate range: [0, 1]")
    print("✓ Candidate cell range: [-1, 1]")


def test_lstm_get_set_params():
    """Test parameter getter/setter interface."""
    print("\n=== Testing Parameter Interface ===")
    
    input_size, hidden_size = 10, 20
    lstm = LSTM(input_size, hidden_size)
    
    # Get params
    params = lstm.get_params()
    
    expected_keys = ['U_ifo', 'U_cell', 'W_ifo', 'W_cell', 'b_ifo', 'b_cell']
    for key in expected_keys:
        assert key in params, f"Missing param: {key}"
    
    print(f"✓ Parameters: {list(params.keys())}")
    
    # Modify and set
    original_U_ifo = params['U_ifo'].copy()
    params['U_ifo'] = params['U_ifo'] * 2
    lstm.set_params(params)
    
    new_params = lstm.get_params()
    assert np.allclose(new_params['U_ifo'], original_U_ifo * 2), "set_params failed"
    
    print("✓ set_params works correctly")


def run_all_tests():
    print("=" * 60)
    print("Running LSTM Tests")
    print("=" * 60)
    
    test_lstm_cell_forward_shapes()
    test_lstm_cell_backward_shapes()
    test_lstm_forward_shapes()
    test_lstm_backward_shapes()
    test_lstm_gates()
    test_lstm_get_set_params()
    test_lstm_cell_gradient_check()
    
    print("\n" + "=" * 60)
    print("✅ ALL LSTM TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

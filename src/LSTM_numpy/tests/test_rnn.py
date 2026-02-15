"""
Unit tests for Vanilla RNN Cell

Tests forward pass, backward pass (BPTT), gradient checking,
and demonstrates the vanishing gradient problem.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.RNN_numpy.layers.rnn import RNNCell, RNN


def numerical_gradient_rnn(rnn_cell, x_t, h_prev, grad_h_t, param_name, epsilon=1e-5):
    """
    Compute numerical gradient for RNN cell parameters.
    
    Args:
        rnn_cell: RNNCell instance
        x_t: Input
        h_prev: Previous hidden state
        grad_h_t: Gradient from next layer
        param_name: Which parameter to check ('W_xh', 'W_hh', or 'b_h')
        epsilon: Perturbation size
    """
    param = getattr(rnn_cell, param_name)
    numerical_grad = np.zeros_like(param)
    
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]
        
        # Perturb +epsilon
        param[idx] = old_value + epsilon
        h_pos = rnn_cell.forward(x_t, h_prev)
        loss_pos = np.sum(h_pos * grad_h_t)
        
        # Perturb -epsilon
        param[idx] = old_value - epsilon
        h_neg = rnn_cell.forward(x_t, h_prev)
        loss_neg = np.sum(h_neg * grad_h_t)
        
        # Compute gradient
        numerical_grad[idx] = (loss_pos - loss_neg) / (2 * epsilon)
        
        # Restore
        param[idx] = old_value
        it.iternext()
    
    return numerical_grad


def test_forward_shape():
    """Test that forward pass produces correct shapes."""
    print("\n=== Testing Forward Pass Shapes ===")
    
    batch_size = 4
    input_size = 10
    hidden_size = 20
    
    rnn = RNNCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    
    h_t = rnn.forward(x_t, h_prev)
    
    assert h_t.shape == (batch_size, hidden_size), f"Wrong shape: {h_t.shape}"
    print(f"✓ Output shape correct: {h_t.shape}")


def test_forward_values():
    """Test forward pass with known values."""
    print("\n=== Testing Forward Pass Values ===")
    
    # Small dimensions for manual verification
    rnn = RNNCell(input_size=2, hidden_size=2)
    
    # Set known weights
    rnn.W_xh = np.array([[0.5, -0.3],
                         [-0.1, 0.4]])
    rnn.W_hh = np.array([[0.3, -0.2],
                         [0.1, 0.5]])
    rnn.b_h = np.array([0.1, -0.1])
    
    # Single sample (batch=1)
    x_t = np.array([[1.0, 0.5]])
    h_prev = np.array([[0.3, -0.5]])
    
    h_t = rnn.forward(x_t, h_prev)
    
    # Manual calculation:
    # z = W_xh @ x + W_hh @ h + b
    z_expected = (rnn.W_xh @ x_t.T + rnn.W_hh @ h_prev.T + rnn.b_h.reshape(-1, 1)).flatten()
    h_expected = np.tanh(z_expected).reshape(1, -1)
    
    assert np.allclose(h_t, h_expected, atol=1e-6), \
        f"Forward values incorrect:\n{h_t}\nvs\n{h_expected}"
    print("✓ Forward pass computation correct")
    print(f"  Hidden state range: [{h_t.min():.3f}, {h_t.max():.3f}]")


def test_backward_shapes():
    """Test that backward pass produces correct gradient shapes."""
    print("\n=== Testing Backward Pass Shapes ===")
    
    batch_size = 4
    input_size = 10
    hidden_size = 20
    
    rnn = RNNCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    
    # Forward
    h_t = rnn.forward(x_t, h_prev)
    
    # Backward
    grad_h_t = np.random.randn(batch_size, hidden_size)
    grad_x_t, grad_h_prev = rnn.backward(grad_h_t)
    
    # Check shapes
    assert grad_x_t.shape == (batch_size, input_size), f"grad_x_t shape wrong: {grad_x_t.shape}"
    assert grad_h_prev.shape == (batch_size, hidden_size), f"grad_h_prev shape wrong: {grad_h_prev.shape}"
    
    assert rnn.grad_W_xh.shape == (hidden_size, input_size), f"grad_W_xh shape wrong"
    assert rnn.grad_W_hh.shape == (hidden_size, hidden_size), f"grad_W_hh shape wrong"
    assert rnn.grad_b_h.shape == (hidden_size,), f"grad_b_h shape wrong"
    
    print("✓ All gradient shapes correct")


def test_gradient_checking_W_xh():
    """Validate grad_W_xh using numerical gradient."""
    print("\n=== Gradient Checking: W_xh ===")
    
    np.random.seed(42)
    rnn = RNNCell(input_size=5, hidden_size=4)
    
    x_t = np.random.randn(2, 5)
    h_prev = np.random.randn(2, 4)
    
    # Forward
    h_t = rnn.forward(x_t, h_prev)
    
    # Backward
    grad_h_t = np.random.randn(2, 4)
    grad_x_t, grad_h_prev = rnn.backward(grad_h_t)
    
    # Analytical gradient
    analytical = rnn.grad_W_xh
    
    # Numerical gradient
    numerical = numerical_gradient_rnn(rnn, x_t, h_prev, grad_h_t, 'W_xh')
    
    # Compare
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Gradient check failed for W_xh: {max_error}"
    print("✓ W_xh gradient correct")


def test_gradient_checking_W_hh():
    """Validate grad_W_hh using numerical gradient."""
    print("\n=== Gradient Checking: W_hh ===")
    
    np.random.seed(42)
    rnn = RNNCell(input_size=5, hidden_size=4)
    
    x_t = np.random.randn(2, 5)
    h_prev = np.random.randn(2, 4)
    
    # Forward
    h_t = rnn.forward(x_t, h_prev)
    
    # Backward
    grad_h_t = np.random.randn(2, 4)
    grad_x_t, grad_h_prev = rnn.backward(grad_h_t)
    
    # Analytical gradient
    analytical = rnn.grad_W_hh
    
    # Numerical gradient
    numerical = numerical_gradient_rnn(rnn, x_t, h_prev, grad_h_t, 'W_hh')
    
    # Compare
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Gradient check failed for W_hh: {max_error}"
    print("✓ W_hh gradient correct")


def test_gradient_checking_b_h():
    """Validate grad_b_h using numerical gradient."""
    print("\n=== Gradient Checking: b_h ===")
    
    np.random.seed(42)
    rnn = RNNCell(input_size=5, hidden_size=4)
    
    x_t = np.random.randn(2, 5)
    h_prev = np.random.randn(2, 4)
    
    # Forward
    h_t = rnn.forward(x_t, h_prev)
    
    # Backward
    grad_h_t = np.random.randn(2, 4)
    grad_x_t, grad_h_prev = rnn.backward(grad_h_t)
    
    # Analytical gradient
    analytical = rnn.grad_b_h
    
    # Numerical gradient
    numerical = numerical_gradient_rnn(rnn, x_t, h_prev, grad_h_t, 'b_h')
    
    # Compare
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Gradient check failed for b_h: {max_error}"
    print("✓ b_h gradient correct")


def test_vanishing_gradients():
    """Demonstrate the vanishing gradient problem."""
    print("\n=== Demonstrating Vanishing Gradient Problem ===")
    
    seq_lengths = [5, 10, 20, 50, 100]
    input_size = 10
    hidden_size = 20
    batch_size = 1
    
    print("\nGradient magnitudes as sequence length increases:")
    print("(This shows why vanilla RNN struggles with long sequences)\n")
    
    for seq_len in seq_lengths:
        rnn = RNN(input_size, hidden_size)
        
        # Random sequence
        x_seq = np.random.randn(batch_size, seq_len, input_size) * 0.1
        
        # Forward
        h_seq, h_final = rnn.forward(x_seq)
        
        # Backward with gradient only at final timestep
        grad_h_seq = np.zeros_like(h_seq)
        grad_h_seq[:, -1, :] = np.ones((batch_size, hidden_size))
        
        grad_x_seq = rnn.backward(grad_h_seq)
        
        # Measure gradient magnitude at first timestep
        grad_norm_first = np.linalg.norm(grad_x_seq[:, 0, :])
        grad_norm_last = np.linalg.norm(grad_x_seq[:, -1, :])
        
        print(f"  Seq length {seq_len:3d}: "
              f"grad_norm(first)={grad_norm_first:8.6f}, "
              f"grad_norm(last)={grad_norm_last:8.6f}, "
              f"ratio={grad_norm_first/grad_norm_last:8.6f}")
    
    print("\n⚠ Notice: Gradients decay exponentially with sequence length!")
    print("  This is the vanishing gradient problem.")
    print("  LSTM solves this with gates that preserve gradient flow.")


def test_full_sequence():
    """Test RNN class on a full sequence."""
    print("\n=== Testing Full Sequence Processing ===")
    
    batch_size = 3
    seq_len = 10
    input_size = 8
    hidden_size = 16
    
    rnn = RNN(input_size, hidden_size)
    
    # Random sequence
    x_seq = np.random.randn(batch_size, seq_len, input_size)
    
    # Forward
    h_seq, h_final = rnn.forward(x_seq)
    
    assert h_seq.shape == (batch_size, seq_len, hidden_size), f"Wrong h_seq shape: {h_seq.shape}"
    assert h_final.shape == (batch_size, hidden_size), f"Wrong h_final shape: {h_final.shape}"
    
    # Check that h_final matches last timestep
    assert np.allclose(h_final, h_seq[:, -1, :]), "h_final should match last timestep"
    
    print(f"✓ Sequence processing correct")
    print(f"  Input: {x_seq.shape}")
    print(f"  Hidden states: {h_seq.shape}")
    print(f"  Final state: {h_final.shape}")


def run_all_tests():
    """Run all RNN tests."""
    print("=" * 60)
    print("Running Vanilla RNN Tests")
    print("=" * 60)
    
    test_forward_shape()
    test_forward_values()
    test_backward_shapes()
    test_gradient_checking_W_xh()
    test_gradient_checking_W_hh()
    test_gradient_checking_b_h()
    test_full_sequence()
    test_vanishing_gradients()
    
    print("\n" + "=" * 60)
    print("✅ ALL RNN TESTS PASSED!")
    print("=" * 60)
    print("\nKey Learnings:")
    print("1. RNN maintains hidden state across timesteps")
    print("2. BPTT correctly computes gradients through time")
    print("3. Vanishing gradients occur in long sequences")
    print("4. This motivates LSTM's gating mechanism")
    print("\nNext: Implement LSTM to solve vanishing gradient problem!")


if __name__ == "__main__":
    run_all_tests()

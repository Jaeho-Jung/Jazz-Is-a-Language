"""
Unit tests for Linear (fully connected) layer.

Tests forward pass, backward pass, and gradient checking.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.RNN_numpy.layers.linear import Linear


def numerical_gradient_linear(linear, x, grad_output, param_name, epsilon=1e-5):
    """
    Compute numerical gradient for Linear layer parameters.
    
    Args:
        linear: Linear layer instance
        x: Input
        grad_output: Gradient from next layer
        param_name: Which parameter to check ('W' or 'b')
        epsilon: Perturbation size
    """
    param = getattr(linear, param_name)
    numerical_grad = np.zeros_like(param)
    
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]
        
        # Perturb +epsilon
        param[idx] = old_value + epsilon
        output_pos = linear.forward(x)
        loss_pos = np.sum(output_pos * grad_output)
        
        # Perturb -epsilon
        param[idx] = old_value - epsilon
        output_neg = linear.forward(x)
        loss_neg = np.sum(output_neg * grad_output)
        
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
    input_features = 10
    output_features = 5
    
    linear = Linear(input_features, output_features)
    x = np.random.randn(batch_size, input_features)
    
    y = linear.forward(x)
    
    assert y.shape == (batch_size, output_features), f"Wrong shape: {y.shape}"
    print(f"✓ Output shape correct: {y.shape}")


def test_forward_values():
    """Test forward pass with known values."""
    print("\n=== Testing Forward Pass Values ===")
    
    # Small dimensions for manual verification
    linear = Linear(input_features=3, output_features=2)
    
    # Set known weights
    linear.W = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    linear.b = np.array([0.1, 0.2])
    
    # Single sample (batch=1)
    x = np.array([[1.0, 0.5, -0.2]])
    
    # Manual calculation: y = x @ W.T + b
    # W.T = [[1.0, 4.0],    x = [1.0, 0.5, -0.2]
    #        [2.0, 5.0],
    #        [3.0, 6.0]]
    # x @ W.T = [1*1 + 0.5*2 + (-0.2)*3, 1*4 + 0.5*5 + (-0.2)*6]
    #         = [1 + 1 - 0.6, 4 + 2.5 - 1.2]
    #         = [1.4, 5.3]
    # + b = [1.4 + 0.1, 5.3 + 0.2] = [1.5, 5.5]
    
    y = linear.forward(x)
    expected = np.array([[1.5, 5.5]])
    
    assert np.allclose(y, expected, atol=1e-6), \
        f"Forward values incorrect:\n{y}\nvs\n{expected}"
    print("✓ Forward pass computation correct")
    print(f"  Output: {y[0]}")


def test_backward_shapes():
    """Test that backward pass produces correct gradient shapes."""
    print("\n=== Testing Backward Pass Shapes ===")
    
    batch_size = 4
    input_features = 10
    output_features = 5
    
    linear = Linear(input_features, output_features)
    x = np.random.randn(batch_size, input_features)
    
    # Forward
    y = linear.forward(x)
    
    # Backward
    grad_output = np.random.randn(batch_size, output_features)
    grad_x = linear.backward(grad_output)
    
    # Check shapes
    assert grad_x.shape == (batch_size, input_features), \
        f"grad_x shape wrong: {grad_x.shape}"
    assert linear.grad_W.shape == (output_features, input_features), \
        f"grad_W shape wrong: {linear.grad_W.shape}"
    assert linear.grad_b.shape == (output_features,), \
        f"grad_b shape wrong: {linear.grad_b.shape}"
    
    print("✓ All gradient shapes correct")


def test_gradient_checking_W():
    """Validate grad_W using numerical gradient."""
    print("\n=== Gradient Checking: W ===")
    
    np.random.seed(42)
    linear = Linear(input_features=5, output_features=3)
    
    x = np.random.randn(2, 5)
    
    # Forward
    y = linear.forward(x)
    
    # Backward
    grad_output = np.random.randn(2, 3)
    grad_x = linear.backward(grad_output)
    
    # Analytical gradient
    analytical = linear.grad_W
    
    # Numerical gradient
    numerical = numerical_gradient_linear(linear, x, grad_output, 'W')
    
    # Compare
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Gradient check failed for W: {max_error}"
    print("✓ W gradient correct")


def test_gradient_checking_b():
    """Validate grad_b using numerical gradient."""
    print("\n=== Gradient Checking: b ===")
    
    np.random.seed(42)
    linear = Linear(input_features=5, output_features=3)
    
    x = np.random.randn(2, 5)
    
    # Forward
    y = linear.forward(x)
    
    # Backward
    grad_output = np.random.randn(2, 3)
    grad_x = linear.backward(grad_output)
    
    # Analytical gradient
    analytical = linear.grad_b
    
    # Numerical gradient
    numerical = numerical_gradient_linear(linear, x, grad_output, 'b')
    
    # Compare
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Gradient check failed for b: {max_error}"
    print("✓ b gradient correct")


def test_batched_gradients():
    """Test that gradients are correctly summed across batch."""
    print("\n=== Testing Batched Gradient Accumulation ===")
    
    linear = Linear(input_features=3, output_features=2)
    
    # Manually set weights
    linear.W = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    linear.b = np.array([0.1, 0.2])
    
    # Batch of 2 samples
    x = np.array([[1.0, 0.5, -0.2],
                  [0.3, -0.1, 0.8]])
    
    # Forward
    y = linear.forward(x)
    
    # Backward with known gradients
    grad_output = np.array([[1.0, 2.0],
                            [3.0, 4.0]])
    
    grad_x = linear.backward(grad_output)
    
    # Check bias gradient (should sum across batch)
    expected_grad_b = np.sum(grad_output, axis=0)  # [1+3, 2+4] = [4, 6]
    assert np.allclose(linear.grad_b, expected_grad_b), \
        f"Bias gradient incorrect: {linear.grad_b} vs {expected_grad_b}"
    
    print("✓ Batch gradient accumulation correct")
    print(f"  grad_b: {linear.grad_b}")


def test_comparison_with_pytorch():
    """Compare with PyTorch Linear layer if available."""
    try:
        import torch
        import torch.nn as nn
        
        print("\n=== Comparing with PyTorch ===")
        
        # Create both layers
        input_features, output_features = 10, 5
        np_linear = Linear(input_features, output_features)
        torch_linear = nn.Linear(input_features, output_features)
        
        # Use same weights
        torch_linear.weight.data = torch.from_numpy(np_linear.W).float()
        torch_linear.bias.data = torch.from_numpy(np_linear.b).float()
        
        # Test forward pass
        x_np = np.random.randn(4, input_features)
        x_torch = torch.from_numpy(x_np).float()
        
        y_np = np_linear.forward(x_np)
        y_torch = torch_linear(x_torch).detach().numpy()
        
        assert np.allclose(y_np, y_torch, atol=1e-6), "Forward doesn't match PyTorch"
        print("✓ Forward pass matches PyTorch")
        
        # Test backward pass
        grad_out_np = np.random.randn(4, output_features)
        grad_out_torch = torch.from_numpy(grad_out_np).float()
        
        # NumPy backward
        grad_x_np = np_linear.backward(grad_out_np)
        
        # PyTorch backward
        y_torch.backward(grad_out_torch)
        grad_w_torch = torch_linear.weight.grad.numpy()
        grad_b_torch = torch_linear.bias.grad.numpy()
        
        assert np.allclose(np_linear.grad_W, grad_w_torch, atol=1e-6), \
            "W gradient doesn't match PyTorch"
        assert np.allclose(np_linear.grad_b, grad_b_torch, atol=1e-6), \
            "b gradient doesn't match PyTorch"
        
        print("✓ Backward pass matches PyTorch")
        
    except ImportError:
        print("\n⚠ PyTorch not available, skipping comparison tests")


def run_all_tests():
    """Run all Linear layer tests."""
    print("=" * 60)
    print("Running Linear Layer Tests")
    print("=" * 60)
    
    test_forward_shape()
    test_forward_values()
    test_backward_shapes()
    test_gradient_checking_W()
    test_gradient_checking_b()
    test_batched_gradients()
    test_comparison_with_pytorch()
    
    print("\n" + "=" * 60)
    print("✅ ALL LINEAR LAYER TESTS PASSED!")
    print("=" * 60)
    print("\nKey Learnings:")
    print("1. Linear layer: y = xW^T + b")
    print("2. Weight gradient: ∂L/∂W = ∂L/∂y^T @ x")
    print("3. Bias gradient: ∂L/∂b = sum(∂L/∂y, axis=0)")
    print("4. Input gradient: ∂L/∂x = ∂L/∂y @ W")


if __name__ == "__main__":
    run_all_tests()

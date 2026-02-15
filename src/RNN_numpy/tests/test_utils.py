"""
Unit tests for NumPy RNN utilities.

Tests activation functions, softmax, cross-entropy loss, and derivatives
using gradient checking and comparison with PyTorch.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.RNN_numpy.utils import (
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    softmax, cross_entropy_loss, cross_entropy_grad
)


def numerical_gradient(f, x, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences.
    This is the gold standard for validating analytical gradients.
    
    f: scalar function taking array input
    x: input array
    epsilon: small perturbation for finite difference
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + epsilon
        pos = f(x.copy())
        
        x[idx] = old_value - epsilon
        neg = f(x.copy())
        
        grad[idx] = (pos - neg) / (2 * epsilon)
        x[idx] = old_value
        it.iternext()
    
    return grad


def test_sigmoid():
    """Test sigmoid function against known values."""
    print("\n=== Testing Sigmoid ===")
    
    # Test case 1: Known values
    assert np.isclose(sigmoid(0), 0.5), "sigmoid(0) should be 0.5"
    assert np.isclose(sigmoid(1000), 1.0, atol=1e-6), "sigmoid(large) should approach 1"
    assert np.isclose(sigmoid(-1000), 0.0, atol=1e-6), "sigmoid(-large) should approach 0"
    
    # Test case 2: Array input
    z = np.array([-2, -1, 0, 1, 2])
    expected = 1 / (1 + np.exp(-z))
    result = sigmoid(z)
    assert np.allclose(result, expected), "Sigmoid array computation incorrect"
    
    print("✓ Sigmoid values correct")
    
    # Test case 3: Gradient checking
    test_point = np.random.randn(5)
    
    def scalar_sigmoid(x):
        return np.sum(sigmoid(x))  # Sum to get scalar output
    
    analytical = sigmoid_derivative(test_point)
    numerical = numerical_gradient(scalar_sigmoid, test_point)
    
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    assert max_error < 1e-5, f"Sigmoid derivative gradient check failed: {max_error}"
    print("✓ Sigmoid derivative correct (gradient check passed)")


def test_tanh():
    """Test tanh function and its derivative."""
    print("\n=== Testing Tanh ===")
    
    # Test case 1: Known values
    assert np.isclose(tanh(0), 0.0), "tanh(0) should be 0"
    assert np.isclose(tanh(1000), 1.0, atol=1e-6), "tanh(large) should approach 1"
    assert np.isclose(tanh(-1000), -1.0, atol=1e-6), "tanh(-large) should approach -1"
    
    # Test case 2: Compare with NumPy
    z = np.random.randn(10)
    assert np.allclose(tanh(z), np.tanh(z)), "Tanh should match np.tanh"
    
    print("✓ Tanh values correct")
    
    # Test case 3: Gradient checking
    test_point = np.random.randn(5)
    
    def scalar_tanh(x):
        return np.sum(tanh(x))
    
    analytical = tanh_derivative(test_point)
    numerical = numerical_gradient(scalar_tanh, test_point)
    
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    assert max_error < 1e-5, f"Tanh derivative gradient check failed: {max_error}"
    print("✓ Tanh derivative correct (gradient check passed)")


def test_softmax_numerical_stability():
    """Test softmax numerical stability with large values."""
    print("\n=== Testing Softmax Numerical Stability ===")
    
    # Test case 1: Normal values
    z = np.array([[1, 2, 3], [4, 5, 6]])
    probs = softmax(z)
    
    # Check probabilities sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Softmax should sum to 1"
    print("✓ Softmax sums to 1")
    
    # Check all probabilities are positive
    assert np.all(probs > 0) and np.all(probs < 1), "Softmax should be in (0, 1)"
    print("✓ Softmax values in (0, 1)")
    
    # Test case 2: Large values (numerical stability test)
    z_large = np.array([[1000, 1001, 1002], [500, 501, 502]])
    probs_large = softmax(z_large)
    
    # Should not overflow to inf or nan
    assert not np.any(np.isnan(probs_large)), "Softmax produced NaN with large values"
    assert not np.any(np.isinf(probs_large)), "Softmax produced Inf with large values"
    assert np.allclose(np.sum(probs_large, axis=1), 1.0), "Softmax should sum to 1 even with large values"
    print("✓ Softmax numerically stable with large values")
    
    # Test case 3: Compare with naive implementation for normal values
    def naive_softmax(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    z_normal = np.random.randn(5, 10)
    stable_result = softmax(z_normal)
    naive_result = naive_softmax(z_normal)
    
    assert np.allclose(stable_result, naive_result), "Stable softmax should match naive for normal values"
    print("✓ Softmax matches naive implementation for normal values")


def test_cross_entropy():
    """Test cross-entropy loss and gradient."""
    print("\n=== Testing Cross-Entropy Loss ===")
    
    # Test case 1: Perfect prediction
    logits = np.array([[10, 0, 0], [0, 10, 0]])
    targets = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot encoded
    
    loss = cross_entropy_loss(logits, targets)
    print(f"  Loss (perfect prediction): {loss:.6f}")
    assert loss < 0.01, "Perfect prediction should have very low loss"
    print("✓ Cross-entropy loss correct for perfect prediction")
    
    # Test case 2: Random prediction
    logits_random = np.random.randn(10, 5)
    targets_random = np.zeros((10, 5))
    targets_random[np.arange(10), np.random.randint(0, 5, 10)] = 1  # One-hot
    
    loss_random = cross_entropy_loss(logits_random, targets_random)
    print(f"  Loss (random prediction): {loss_random:.6f}")
    assert loss_random > 0, "Cross-entropy loss should be positive"
    print("✓ Cross-entropy loss positive for random prediction")
    
    # Test case 3: Gradient checking
    print("\n  Gradient Checking...")
    
    def loss_fn(logits_flat):
        logits_reshaped = logits_flat.reshape(logits_random.shape)
        return cross_entropy_loss(logits_reshaped, targets_random)
    
    analytical_grad = cross_entropy_grad(logits_random, targets_random)
    numerical_grad = numerical_gradient(loss_fn, logits_random.flatten()).reshape(logits_random.shape)
    
    rel_error = np.abs(analytical_grad - numerical_grad) / (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Cross-entropy gradient check failed: max error {max_error}"
    print("✓ Cross-entropy gradient correct (gradient check passed)")


def test_comparison_with_pytorch():
    """
    Optional: Compare with PyTorch implementations.
    This validates our NumPy code against a trusted reference.
    """
    try:
        import torch
        import torch.nn.functional as F
        
        print("\n=== Comparing with PyTorch ===")
        
        # Test sigmoid
        z_np = np.random.randn(10, 20)
        z_torch = torch.from_numpy(z_np)
        
        sigmoid_np = sigmoid(z_np)
        sigmoid_torch = torch.sigmoid(z_torch).numpy()
        
        assert np.allclose(sigmoid_np, sigmoid_torch, atol=1e-6), "Sigmoid doesn't match PyTorch"
        print("✓ Sigmoid matches PyTorch")
        
        # Test tanh
        tanh_np = tanh(z_np)
        tanh_torch = torch.tanh(z_torch).numpy()
        
        assert np.allclose(tanh_np, tanh_torch, atol=1e-6), "Tanh doesn't match PyTorch"
        print("✓ Tanh matches PyTorch")
        
        # Test softmax
        logits_np = np.random.randn(5, 10)
        logits_torch = torch.from_numpy(logits_np)
        
        softmax_np = softmax(logits_np)
        softmax_torch = F.softmax(logits_torch, dim=1).numpy()
        
        assert np.allclose(softmax_np, softmax_torch, atol=1e-6), "Softmax doesn't match PyTorch"
        print("✓ Softmax matches PyTorch")
        
        # Test cross-entropy
        targets_np = np.zeros((5, 10))
        targets_np[np.arange(5), np.random.randint(0, 10, 5)] = 1
        targets_torch = torch.from_numpy(targets_np)
        
        loss_np = cross_entropy_loss(logits_np, targets_np)
        
        # PyTorch cross-entropy expects class indices, not one-hot
        target_indices = torch.from_numpy(np.argmax(targets_np, axis=1))
        loss_torch = F.cross_entropy(logits_torch, target_indices).item()
        
        print(f"  NumPy loss: {loss_np:.6f}, PyTorch loss: {loss_torch:.6f}")
        assert np.isclose(loss_np, loss_torch, atol=1e-6), "Cross-entropy doesn't match PyTorch"
        print("✓ Cross-entropy matches PyTorch")
        
    except ImportError:
        print("\n⚠ PyTorch not available, skipping comparison tests")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Running NumPy Utils Tests")
    print("=" * 60)
    
    test_sigmoid()
    test_tanh()
    test_softmax_numerical_stability()
    test_cross_entropy()
    test_comparison_with_pytorch()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nKey Learnings:")
    print("1. Activation derivatives validated via gradient checking")
    print("2. Softmax is numerically stable (no overflow with large values)")
    print("3. Cross-entropy gradient is correct (matches numerical gradient)")
    print("4. All implementations match PyTorch (trusted reference)")


if __name__ == "__main__":
    run_all_tests()

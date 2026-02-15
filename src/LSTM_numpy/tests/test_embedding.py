"""
Unit tests for Embedding layer.

Tests forward pass (lookup), backward pass (gradient accumulation),
and validates correctness using gradient checking.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.RNN_numpy.layers.embedding import Embedding


def numerical_gradient_embedding(embed, indices, grad_output, epsilon=1e-5):
    """
    Compute numerical gradient for embedding layer.
    
    We perturb each element of W and measure the change in loss.
    """
    numerical_grad = np.zeros_like(embed.W)
    
    # Flatten for iteration
    it = np.nditer(embed.W, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = embed.W[idx]
        
        # Perturb +epsilon
        embed.W[idx] = old_value + epsilon
        output_pos = embed.W[indices]
        loss_pos = np.sum(output_pos * grad_output)  # Simplified loss
        
        # Perturb -epsilon
        embed.W[idx] = old_value - epsilon
        output_neg = embed.W[indices]
        loss_neg = np.sum(output_neg * grad_output)
        
        # Compute gradient
        numerical_grad[idx] = (loss_pos - loss_neg) / (2 * epsilon)
        
        # Restore
        embed.W[idx] = old_value
        it.iternext()
    
    return numerical_grad


def test_forward_1d():
    """Test forward pass with 1D indices (single batch dimension)."""
    print("\n=== Testing Forward Pass (1D) ===")
    
    # Create small embedding
    embed = Embedding(num_embeddings=5, embedding_dim=3)
    
    # Set known weights for testing
    embed.W = np.array([
        [0.1, 0.2, 0.3],  # Index 0
        [0.4, 0.5, 0.6],  # Index 1
        [0.7, 0.8, 0.9],  # Index 2
        [1.0, 1.1, 1.2],  # Index 3
        [1.3, 1.4, 1.5],  # Index 4
    ])
    
    # Test lookup
    indices = np.array([0, 2, 4])
    output = embed.forward(indices)
    
    expected = np.array([
        [0.1, 0.2, 0.3],
        [0.7, 0.8, 0.9],
        [1.3, 1.4, 1.5],
    ])
    
    assert output.shape == (3, 3), f"Wrong shape: {output.shape}"
    assert np.allclose(output, expected), f"Forward pass incorrect:\n{output}\nvs\n{expected}"
    print("✓ Forward pass (1D) correct")


def test_forward_2d():
    """Test forward pass with 2D indices (batch × seq_len)."""
    print("\n=== Testing Forward Pass (2D) ===")
    
    embed = Embedding(num_embeddings=4, embedding_dim=2)
    embed.W = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])
    
    # 2D indices: batch_size=2, seq_len=3
    indices = np.array([
        [0, 1, 2],
        [2, 3, 0],
    ])
    
    output = embed.forward(indices)
    
    expected = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
    ])
    
    assert output.shape == (2, 3, 2), f"Wrong shape: {output.shape}"
    assert np.allclose(output, expected), "Forward pass (2D) incorrect"
    print("✓ Forward pass (2D) correct")


def test_backward_no_duplicates():
    """Test backward pass when indices are unique (no accumulation needed)."""
    print("\n=== Testing Backward Pass (No Duplicates) ===")
    
    embed = Embedding(num_embeddings=4, embedding_dim=2)
    
    indices = np.array([0, 2, 3])
    output = embed.forward(indices)
    
    # Gradient from next layer
    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    
    embed.backward(grad_output)
    
    # Expected: gradients go to the exact indices used
    expected_grad = np.array([
        [1.0, 2.0],  # Index 0
        [0.0, 0.0],  # Index 1 (not used)
        [3.0, 4.0],  # Index 2
        [5.0, 6.0],  # Index 3
    ])
    
    assert np.allclose(embed.grad_W, expected_grad), \
        f"Backward incorrect:\n{embed.grad_W}\nvs\n{expected_grad}"
    print("✓ Backward pass (no duplicates) correct")


def test_backward_with_duplicates():
    """Test backward pass when indices repeat (gradient accumulation)."""
    print("\n=== Testing Backward Pass (With Duplicates) ===")
    
    embed = Embedding(num_embeddings=3, embedding_dim=2)
    
    # Index 0 appears twice, index 1 appears once
    indices = np.array([0, 1, 0])
    output = embed.forward(indices)
    
    grad_output = np.array([
        [1.0, 2.0],  # Gradient for first occurrence of index 0
        [3.0, 4.0],  # Gradient for index 1
        [5.0, 6.0],  # Gradient for second occurrence of index 0
    ])
    
    embed.backward(grad_output)
    
    # Expected: gradients for index 0 should be summed
    expected_grad = np.array([
        [1.0 + 5.0, 2.0 + 6.0],  # Index 0 (accumulated)
        [3.0, 4.0],               # Index 1
        [0.0, 0.0],               # Index 2 (not used)
    ])
    
    assert np.allclose(embed.grad_W, expected_grad), \
        f"Gradient accumulation incorrect:\n{embed.grad_W}\nvs\n{expected_grad}"
    print("✓ Backward pass (with duplicates) correct - gradients accumulated ✅")


def test_gradient_checking():
    """Validate backward pass using numerical gradient checking."""
    print("\n=== Gradient Checking ===")
    
    # Small embedding for faster testing
    embed = Embedding(num_embeddings=5, embedding_dim=3)
    
    # Random indices
    np.random.seed(42)
    indices = np.array([0, 2, 1, 2, 4])  # Includes duplicates (index 2 appears twice)
    
    # Forward pass
    output = embed.forward(indices)
    
    # Random gradient from next layer
    grad_output = np.random.randn(*output.shape)
    
    # Analytical gradient
    embed.backward(grad_output)
    analytical_grad = embed.grad_W.copy()
    
    # Numerical gradient
    numerical_grad = numerical_gradient_embedding(embed, indices, grad_output)
    
    # Compare
    rel_error = np.abs(analytical_grad - numerical_grad) / \
                (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8)
    
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    assert max_error < 1e-5, f"Gradient check failed: max error {max_error}"
    print("✓ Gradient check passed - backward implementation is correct! 🎉")


def test_comparison_with_pytorch():
    """Compare with PyTorch embedding if available."""
    try:
        import torch
        import torch.nn as nn
        
        print("\n=== Comparing with PyTorch ===")
        
        # Create both embeddings
        num_emb, emb_dim = 10, 8
        np_embed = Embedding(num_emb, emb_dim)
        torch_embed = nn.Embedding(num_emb, emb_dim)
        
        # Use same weights
        torch_embed.weight.data = torch.from_numpy(np_embed.W).float()
        
        # Test forward pass
        indices_np = np.array([0, 5, 3, 7, 5])
        indices_torch = torch.from_numpy(indices_np).long()
        
        output_np = np_embed.forward(indices_np)
        output_torch = torch_embed(indices_torch).detach().numpy()
        
        assert np.allclose(output_np, output_torch, atol=1e-6), "Forward doesn't match PyTorch"
        print("✓ Forward pass matches PyTorch")
        
        # Test backward pass
        grad_out_np = np.random.randn(*output_np.shape)
        grad_out_torch = torch.from_numpy(grad_out_np).float()
        
        # NumPy backward
        np_embed.backward(grad_out_np)
        
        # PyTorch backward
        output_torch.backward(grad_out_torch)
        grad_torch = torch_embed.weight.grad.numpy()
        
        assert np.allclose(np_embed.grad_W, grad_torch, atol=1e-6), \
            "Backward doesn't match PyTorch"
        print("✓ Backward pass matches PyTorch")
        
    except ImportError:
        print("\n⚠ PyTorch not available, skipping comparison tests")


def run_all_tests():
    """Run all embedding tests."""
    print("=" * 60)
    print("Running Embedding Layer Tests")
    print("=" * 60)
    
    test_forward_1d()
    test_forward_2d()
    test_backward_no_duplicates()
    test_backward_with_duplicates()
    test_gradient_checking()
    test_comparison_with_pytorch()
    
    print("\n" + "=" * 60)
    print("✅ ALL EMBEDDING TESTS PASSED!")
    print("=" * 60)
    print("\nKey Learnings:")
    print("1. Forward pass is just array indexing (W[indices])")
    print("2. Backward pass accumulates gradients at used indices")
    print("3. Duplicate indices correctly sum their gradients")
    print("4. Gradient checking validates our backward implementation")


if __name__ == "__main__":
    run_all_tests()

"""
test_layers.py
Unit tests for Embedding, Linear, and LayerNorm layers.
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.layers.embedding import Embedding
from src.Transformer_numpy.layers.linear import Linear
from src.Transformer_numpy.layers.layer_norm import LayerNorm


# ============================================================================
# Embedding Tests
# ============================================================================
class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.num_embeddings = 10
        self.embedding_dim = 4
        self.emb = Embedding(self.num_embeddings, self.embedding_dim)

    def test_init_shape(self):
        """Weight matrix should be (num_embeddings, embedding_dim)."""
        self.assertEqual(self.emb.W.shape, (self.num_embeddings, self.embedding_dim))

    def test_forward_2d_shape(self):
        """Forward with (batch, seq_len) indices → (batch, seq_len, embed_dim)."""
        indices = np.array([[0, 1, 2], [3, 4, 5]])  # (2, 3)
        out = self.emb.forward(indices)
        self.assertEqual(out.shape, (2, 3, self.embedding_dim))

    def test_forward_lookup_correctness(self):
        """Embedding lookup should return correct rows of W."""
        indices = np.array([[0, 2]])
        out = self.emb.forward(indices)
        np.testing.assert_array_equal(out[0, 0], self.emb.W[0])
        np.testing.assert_array_equal(out[0, 1], self.emb.W[2])

    def test_forward_out_of_range_raises(self):
        """Out-of-range indices should trigger assertion."""
        with self.assertRaises(AssertionError):
            self.emb.forward(np.array([[self.num_embeddings]]))

    def test_backward_grad_shape(self):
        """Backward should populate grad_W with same shape as W."""
        indices = np.array([[1, 3]])
        self.emb.forward(indices)
        grad_out = np.ones((1, 2, self.embedding_dim))
        self.emb.backward(grad_out)
        self.assertEqual(self.emb.grad_W.shape, self.emb.W.shape)

    def test_backward_grad_accumulation(self):
        """Duplicate indices should accumulate gradients."""
        indices = np.array([[2, 2]])  # same index twice
        self.emb.forward(indices)
        grad_out = np.ones((1, 2, self.embedding_dim))
        self.emb.backward(grad_out)
        # Row 2 should have accumulated gradient = 2 * ones
        np.testing.assert_array_almost_equal(
            self.emb.grad_W[2], np.full(self.embedding_dim, 2.0)
        )
        # Other rows should remain zero
        np.testing.assert_array_almost_equal(
            self.emb.grad_W[0], np.zeros(self.embedding_dim)
        )

    def test_get_set_params_roundtrip(self):
        """get_params → set_params should preserve weights."""
        params = self.emb.get_params()
        new_emb = Embedding(self.num_embeddings, self.embedding_dim)
        new_emb.set_params(params)
        np.testing.assert_array_equal(new_emb.W, self.emb.W)


# ============================================================================
# Linear Tests
# ============================================================================
class TestLinear(unittest.TestCase):
    def setUp(self):
        self.in_features = 8
        self.out_features = 4
        self.linear = Linear(self.in_features, self.out_features)

    def test_init_shapes(self):
        """W should be (out, in), b should be (out,)."""
        self.assertEqual(self.linear.W.shape, (self.out_features, self.in_features))
        self.assertEqual(self.linear.b.shape, (self.out_features,))

    def test_forward_shape(self):
        """Forward: (batch, in) → (batch, out)."""
        x = np.random.randn(3, self.in_features)
        out = self.linear.forward(x)
        self.assertEqual(out.shape, (3, self.out_features))

    def test_forward_math(self):
        """Forward should compute x @ W^T + b."""
        x = np.random.randn(2, self.in_features)
        out = self.linear.forward(x)
        expected = x @ self.linear.W.T + self.linear.b
        np.testing.assert_array_almost_equal(out, expected)

    def test_backward_grad_shapes(self):
        """Backward should produce correct gradient shapes."""
        x = np.random.randn(3, self.in_features)
        self.linear.forward(x)
        grad_out = np.random.randn(3, self.out_features)
        grad_x = self.linear.backward(grad_out)

        self.assertEqual(grad_x.shape, (3, self.in_features))
        self.assertEqual(self.linear.grad_W.shape, self.linear.W.shape)
        self.assertEqual(self.linear.grad_b.shape, self.linear.b.shape)

    def test_backward_numerical_gradient(self):
        """Numerical gradient check for Linear layer."""
        x = np.random.randn(2, self.in_features)
        self.linear.forward(x)
        grad_out = np.random.randn(2, self.out_features)
        grad_x = self.linear.backward(grad_out)

        # Numerical gradient w.r.t. x
        eps = 1e-5
        num_grad_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_plus = x.copy(); x_plus[i, j] += eps
                x_minus = x.copy(); x_minus[i, j] -= eps
                out_plus = x_plus @ self.linear.W.T + self.linear.b
                out_minus = x_minus @ self.linear.W.T + self.linear.b
                num_grad_x[i, j] = np.sum(grad_out * (out_plus - out_minus)) / (2 * eps)

        np.testing.assert_array_almost_equal(grad_x, num_grad_x, decimal=5)

    def test_get_set_params_roundtrip(self):
        """get_params → set_params should preserve weights and biases."""
        params = self.linear.get_params()
        new_linear = Linear(self.in_features, self.out_features)
        new_linear.set_params(params)
        np.testing.assert_array_equal(new_linear.W, self.linear.W)
        np.testing.assert_array_equal(new_linear.b, self.linear.b)


# ============================================================================
# LayerNorm Tests
# ============================================================================
class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.normalized_shape = 8
        self.ln = LayerNorm(self.normalized_shape)

    def test_init_params(self):
        """gamma should be ones, beta should be zeros."""
        np.testing.assert_array_equal(self.ln.gamma, np.ones(self.normalized_shape))
        np.testing.assert_array_equal(self.ln.beta, np.zeros(self.normalized_shape))

    def test_forward_shape_2d(self):
        """Forward: (batch, features) → same shape."""
        x = np.random.randn(4, self.normalized_shape)
        out = self.ln.forward(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_shape_3d(self):
        """Forward: (batch, seq_len, features) → same shape."""
        x = np.random.randn(2, 5, self.normalized_shape)
        out = self.ln.forward(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_normalized_output(self):
        """With default gamma=1, beta=0, output should have mean≈0 and var≈1."""
        x = np.random.randn(4, self.normalized_shape) * 5 + 3  # shifted & scaled
        out = self.ln.forward(x)
        # Check per-sample mean ≈ 0 and var ≈ 1
        means = np.mean(out, axis=-1)
        vars_ = np.var(out, axis=-1)
        np.testing.assert_array_almost_equal(means, 0, decimal=5)
        np.testing.assert_array_almost_equal(vars_, 1, decimal=3)

    def test_forward_gamma_beta_effect(self):
        """Setting gamma and beta should scale and shift output."""
        self.ln.gamma = np.full(self.normalized_shape, 2.0)
        self.ln.beta = np.full(self.normalized_shape, 1.0)
        x = np.random.randn(4, self.normalized_shape)
        out = self.ln.forward(x)
        # mean should be ≈ beta=1, std should be ≈ gamma=2
        means = np.mean(out, axis=-1)
        stds = np.std(out, axis=-1)
        np.testing.assert_array_almost_equal(means, 1.0, decimal=4)
        np.testing.assert_array_almost_equal(stds, 2.0, decimal=2)

    def test_backward_grad_shapes(self):
        """Backward should produce correct gradient shapes."""
        x = np.random.randn(3, self.normalized_shape)
        self.ln.forward(x)
        grad_out = np.random.randn(3, self.normalized_shape)
        grad_x = self.ln.backward(grad_out)

        self.assertEqual(grad_x.shape, x.shape)
        self.assertEqual(self.ln.grad_gamma.shape, (self.normalized_shape,))
        self.assertEqual(self.ln.grad_beta.shape, (self.normalized_shape,))

    def test_backward_numerical_gradient(self):
        """Numerical gradient check for LayerNorm (w.r.t. input)."""
        x = np.random.randn(2, self.normalized_shape)
        self.ln.forward(x)
        grad_out = np.random.randn(2, self.normalized_shape)
        grad_x = self.ln.backward(grad_out)

        eps = 1e-5
        num_grad_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_plus = x.copy(); x_plus[i, j] += eps
                x_minus = x.copy(); x_minus[i, j] -= eps
                out_plus = self.ln.gamma * ((x_plus - np.mean(x_plus, axis=-1, keepdims=True)) /
                           np.sqrt(np.var(x_plus, axis=-1, keepdims=True) + self.ln.eps)) + self.ln.beta
                out_minus = self.ln.gamma * ((x_minus - np.mean(x_minus, axis=-1, keepdims=True)) /
                            np.sqrt(np.var(x_minus, axis=-1, keepdims=True) + self.ln.eps)) + self.ln.beta
                num_grad_x[i, j] = np.sum(grad_out * (out_plus - out_minus)) / (2 * eps)

        np.testing.assert_array_almost_equal(grad_x, num_grad_x, decimal=5)

    def test_get_set_params_roundtrip(self):
        """get_params → set_params should preserve gamma and beta."""
        self.ln.gamma = np.random.randn(self.normalized_shape)
        self.ln.beta = np.random.randn(self.normalized_shape)
        params = self.ln.get_params()
        new_ln = LayerNorm(self.normalized_shape)
        new_ln.set_params(params)
        np.testing.assert_array_equal(new_ln.gamma, self.ln.gamma)
        np.testing.assert_array_equal(new_ln.beta, self.ln.beta)


if __name__ == '__main__':
    unittest.main()

"""
test_transformer.py
Unit tests for MultiHeadCausalSelfAttention, TransformerBlock, and Transformer backbone.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.layers.transformer import (
    MultiHeadCausalSelfAttention,
    TransformerBlock,
    Transformer,
)


class TestMultiHeadCausalSelfAttention(unittest.TestCase):

    def setUp(self):
        self.B, self.T, self.C = 2, 5, 16
        self.n_heads = 4
        self.mha = MultiHeadCausalSelfAttention(
            embed_dim=self.C, num_heads=self.n_heads, dropout_rate=0.0, max_seq_len=32)
        self.x = np.random.randn(self.B, self.T, self.C)

    def test_forward_shape(self):
        out = self.mha.forward(self.x)
        self.assertEqual(out.shape, (self.B, self.T, self.C))

    def test_causal_mask(self):
        """Each position should only attend to positions ≤ itself."""
        attn = self.mha._attn_weights   # populated after forward
        self.mha.forward(self.x)
        attn = self.mha._attn_weights   # (B, h, T, T)
        # Upper triangle (future) should be zero
        upper = np.triu(np.ones((self.T, self.T)), k=1).astype(bool)
        self.assertTrue(np.allclose(attn[:, :, upper], 0.0))

    def test_backward_shape(self):
        self.mha.forward(self.x)
        grad_out = np.random.randn(self.B, self.T, self.C)
        grad_in = self.mha.backward(grad_out)
        self.assertEqual(grad_in.shape, (self.B, self.T, self.C))

    def test_backward_populates_grads(self):
        self.mha.forward(self.x)
        self.mha.backward(np.ones((self.B, self.T, self.C)))
        for proj_name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            proj = getattr(self.mha, proj_name)
            self.assertIsNotNone(proj.grad_W, f"{proj_name}.grad_W is None")
            self.assertIsNotNone(proj.grad_b, f"{proj_name}.grad_b is None")

    def test_get_set_params_roundtrip(self):
        params = self.mha.get_params()
        original_q_W = params['q_proj']['W'].copy()
        mha2 = MultiHeadCausalSelfAttention(self.C, self.n_heads, 0.0, 32)
        mha2.set_params(params)
        np.testing.assert_array_equal(mha2.q_proj.W, original_q_W)


class TestTransformerBlock(unittest.TestCase):

    def setUp(self):
        self.B, self.T, self.C = 2, 5, 16
        self.block = TransformerBlock(embed_dim=self.C, num_heads=4, dropout_rate=0.0, max_seq_len=32)
        self.x = np.random.randn(self.B, self.T, self.C)

    def test_forward_shape(self):
        out = self.block.forward(self.x)
        self.assertEqual(out.shape, (self.B, self.T, self.C))

    def test_backward_shape(self):
        self.block.forward(self.x)
        grad = self.block.backward(np.random.randn(self.B, self.T, self.C))
        self.assertEqual(grad.shape, (self.B, self.T, self.C))

    def test_backward_numerical_gradient(self):
        """Finite-difference check on a single scalar output."""
        eps = 1e-4
        out = self.block.forward(self.x)
        scalar = out.sum()

        grad_out = np.ones_like(out)
        grad_in = self.block.backward(grad_out)

        # Perturb one element and check finite difference vs grad_in
        i, j, k = 0, 2, 3
        x_plus = self.x.copy(); x_plus[i, j, k] += eps
        x_minus = self.x.copy(); x_minus[i, j, k] -= eps

        # Need fresh blocks (no cached state); just check sign consistency
        block2 = TransformerBlock(self.C, 4, 0.0, 32)
        block2.set_params(self.block.get_params())
        out_plus  = block2.forward(x_plus).sum()
        block2.set_params(self.block.get_params())
        out_minus = block2.forward(x_minus).sum()

        fd_grad = (out_plus - out_minus) / (2 * eps)
        analytic = grad_in[i, j, k]
        self.assertAlmostEqual(fd_grad, analytic, delta=1e-2,
                               msg=f"Numerical grad {fd_grad:.4f} vs analytic {analytic:.4f}")


class TestTransformerBackbone(unittest.TestCase):

    def setUp(self):
        self.B, self.T = 2, 5
        self.input_size  = 76
        self.hidden_size = 32
        self.transformer = Transformer(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_heads=4, num_blocks=2, dropout_rate=0.0, max_seq_len=32)
        self.x = np.random.randn(self.B, self.T, self.input_size)

    def test_forward_shape(self):
        out = self.transformer.forward(self.x)
        self.assertEqual(out.shape, (self.B, self.T, self.hidden_size))

    def test_backward_shape(self):
        self.transformer.forward(self.x)
        grad_h = np.random.randn(self.B, self.T, self.hidden_size)
        grad_x = self.transformer.backward(grad_h)
        self.assertEqual(grad_x.shape, (self.B, self.T, self.input_size))

    def test_get_params_has_all_keys(self):
        params = self.transformer.get_params()
        self.assertIn('input_proj', params)
        for i in range(2):
            self.assertIn(f'block_{i}', params)

    def test_set_params_roundtrip(self):
        params = self.transformer.get_params()
        orig_W = params['input_proj']['W'].copy()
        t2 = Transformer(self.input_size, self.hidden_size, 4, 2, 0.0, 32)
        t2.set_params(params)
        np.testing.assert_array_equal(t2.input_proj.W, orig_W)


if __name__ == '__main__':
    unittest.main()

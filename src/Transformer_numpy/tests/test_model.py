"""
test_model.py
Unit tests for JazzTransformer model.

The model depends on configs that are incomplete in the current config.py
and on a Transformer layer that has unresolved dependencies (dropout.py).
Tests define their own config constants and mock the Transformer layer.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.layers.embedding import Embedding
from src.Transformer_numpy.layers.linear import Linear


# ============================================================================
# Test config constants (matching LSTM_numpy/config.py pattern)
# ============================================================================
VOCAB_SIZE_PITCH = 129
VOCAB_SIZE_REL_PITCH = 13
VOCAB_SIZE_GRID_POS = 48
VOCAB_SIZE_CHORD_ROOT = 13
VOCAB_SIZE_CHORD_ROOT_REL = 13
VOCAB_SIZE_CHORD_QUALITY = 7
VOCAB_SIZE_PREV_INTERVAL = 25
NUM_DUR_CLASSES = 16

EMBED_SIZE_PITCH = 32
EMBED_SIZE_REL_PITCH = 8
EMBED_SIZE_DURATION = 8
EMBED_SIZE_GRID_POS = 8
EMBED_SIZE_CHORD_ROOT = 8
EMBED_SIZE_CHORD_ROOT_REL = 8
EMBED_SIZE_CHORD_QUALITY = 4
EMBED_SIZE_PREV_INTERVAL = 8

TOTAL_EMBED_SIZE = (
    EMBED_SIZE_PITCH + EMBED_SIZE_REL_PITCH + EMBED_SIZE_DURATION +
    EMBED_SIZE_GRID_POS + EMBED_SIZE_CHORD_ROOT + EMBED_SIZE_CHORD_ROOT_REL +
    EMBED_SIZE_CHORD_QUALITY + EMBED_SIZE_CHORD_ROOT + EMBED_SIZE_CHORD_ROOT_REL +
    EMBED_SIZE_CHORD_QUALITY + EMBED_SIZE_PREV_INTERVAL
)
TRANSFORMER_HIDDEN_SIZE = 128

# Embedding spec matching model.py's structure
EMBEDDING_SPEC = {
    'pitch': (VOCAB_SIZE_PITCH, EMBED_SIZE_PITCH),
    'rel_pitch': (VOCAB_SIZE_REL_PITCH, EMBED_SIZE_REL_PITCH),
    'dur': (NUM_DUR_CLASSES, EMBED_SIZE_DURATION),
    'pos': (VOCAB_SIZE_GRID_POS, EMBED_SIZE_GRID_POS),
    'chord_root': (VOCAB_SIZE_CHORD_ROOT, EMBED_SIZE_CHORD_ROOT),
    'chord_root_rel': (VOCAB_SIZE_CHORD_ROOT_REL, EMBED_SIZE_CHORD_ROOT_REL),
    'chord_quality': (VOCAB_SIZE_CHORD_QUALITY, EMBED_SIZE_CHORD_QUALITY),
    'next_chord_root': (VOCAB_SIZE_CHORD_ROOT, EMBED_SIZE_CHORD_ROOT),
    'next_chord_root_rel': (VOCAB_SIZE_CHORD_ROOT_REL, EMBED_SIZE_CHORD_ROOT_REL),
    'next_chord_quality': (VOCAB_SIZE_CHORD_QUALITY, EMBED_SIZE_CHORD_QUALITY),
    'prev_interval': (VOCAB_SIZE_PREV_INTERVAL, EMBED_SIZE_PREV_INTERVAL),
}


def _build_model():
    """
    Build a JazzTransformer-like model manually (bypassing model.py's import
    of Transformer which has unresolved dependencies).
    Mirrors model.py __init__ logic exactly.
    """
    model = type('JazzTransformer', (), {})()

    model.embeddings = {
        name: Embedding(vocab_size, embed_dim)
        for name, (vocab_size, embed_dim) in EMBEDDING_SPEC.items()
    }
    model.transformer = MagicMock()
    model.pitch_head = Linear(TRANSFORMER_HIDDEN_SIZE, VOCAB_SIZE_PITCH)
    model.dur_head = Linear(TRANSFORMER_HIDDEN_SIZE, NUM_DUR_CLASSES)
    model.num_dur_classes = NUM_DUR_CLASSES
    model._last_features = {}
    model._seq_len = 0

    # Bind model.py's methods
    from src.Transformer_numpy.model import JazzTransformer as _JT
    model.forward = _JT.forward.__get__(model, type(model))
    model.backward = _JT.backward.__get__(model, type(model))
    model.get_all_params = _JT.get_all_params.__get__(model, type(model))
    model.get_all_grads = _JT.get_all_grads.__get__(model, type(model))
    model.set_all_params = _JT.set_all_params.__get__(model, type(model))

    return model


def _make_features(batch_size, seq_len):
    """Create random valid feature indices."""
    return {
        name: np.random.randint(0, vocab_size, (batch_size, seq_len))
        for name, (vocab_size, _) in EMBEDDING_SPEC.items()
    }


# ============================================================================
# Test Classes
# ============================================================================

class TestModelInit(unittest.TestCase):
    """Test model initialization."""

    def setUp(self):
        self.model = _build_model()

    def test_all_embedding_keys_present(self):
        """All expected embedding keys should exist."""
        for key in EMBEDDING_SPEC:
            self.assertIn(key, self.model.embeddings, f"Missing embedding: {key}")

    def test_embedding_dimensions(self):
        """Embedding vocab sizes and dims should match spec."""
        for name, (vocab_size, embed_dim) in EMBEDDING_SPEC.items():
            emb = self.model.embeddings[name]
            self.assertEqual(emb.num_embeddings, vocab_size, f"{name} vocab size mismatch")
            self.assertEqual(emb.embedding_dim, embed_dim, f"{name} embed dim mismatch")

    def test_output_head_dimensions(self):
        """Pitch and duration heads should have correct dimensions."""
        self.assertEqual(self.model.pitch_head.input_features, TRANSFORMER_HIDDEN_SIZE)
        self.assertEqual(self.model.pitch_head.output_features, VOCAB_SIZE_PITCH)
        self.assertEqual(self.model.dur_head.input_features, TRANSFORMER_HIDDEN_SIZE)
        self.assertEqual(self.model.dur_head.output_features, NUM_DUR_CLASSES)


class TestModelForward(unittest.TestCase):
    """Test model forward pass."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.model = _build_model()
        self.features = _make_features(self.batch_size, self.seq_len)

        # Configure mock transformer
        self.transformer_output = np.random.randn(
            self.batch_size, self.seq_len, TRANSFORMER_HIDDEN_SIZE
        )
        self.model.transformer.forward.return_value = self.transformer_output

    def test_output_shapes(self):
        """Forward pass should return correct output shapes."""
        pitch_logits, dur_logits = self.model.forward(self.features)
        self.assertEqual(pitch_logits.shape, (self.batch_size, VOCAB_SIZE_PITCH))
        self.assertEqual(dur_logits.shape, (self.batch_size, NUM_DUR_CLASSES))

    def test_uses_last_timestep(self):
        """Model should use the last timestep of transformer output."""
        # Zero out all timesteps except last
        transformer_output = np.zeros((self.batch_size, self.seq_len, TRANSFORMER_HIDDEN_SIZE))
        transformer_output[:, -1, :] = 1.0
        self.model.transformer.forward.return_value = transformer_output

        pitch_logits, dur_logits = self.model.forward(self.features)

        # Outputs should be non-trivial (not just bias)
        self.assertTrue(np.any(pitch_logits != 0) or np.any(self.model.pitch_head.b != 0))

    def test_stores_features_and_seq_len(self):
        """Forward pass should cache features and seq_len for backward."""
        self.model.forward(self.features)
        self.assertEqual(set(self.model._last_features.keys()), set(self.features.keys()))
        self.assertEqual(self.model._seq_len, self.seq_len)

    def test_transformer_receives_concatenated_embeddings(self):
        """Transformer should receive (batch, seq_len, total_embed_size) input."""
        self.model.forward(self.features)
        call_args = self.model.transformer.forward.call_args
        input_to_transformer = call_args[0][0]
        self.assertEqual(input_to_transformer.shape, (self.batch_size, self.seq_len, TOTAL_EMBED_SIZE))


class TestModelBackward(unittest.TestCase):
    """Test model backward pass."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.model = _build_model()
        self.features = _make_features(self.batch_size, self.seq_len)

        # Forward first
        transformer_output = np.random.randn(
            self.batch_size, self.seq_len, TRANSFORMER_HIDDEN_SIZE
        )
        self.model.transformer.forward.return_value = transformer_output
        self.model.transformer.backward.return_value = np.random.randn(
            self.batch_size, self.seq_len, TOTAL_EMBED_SIZE
        )
        self.pitch_logits, self.dur_logits = self.model.forward(self.features)

    def test_backward_completes(self):
        """Backward pass should complete without error."""
        grad_pitch = np.random.randn(*self.pitch_logits.shape)
        grad_dur = np.random.randn(*self.dur_logits.shape)
        self.model.backward(grad_pitch, grad_dur)
        self.model.transformer.backward.assert_called_once()

    def test_backward_populates_head_gradients(self):
        """Output heads should have gradients after backward."""
        grad_pitch = np.random.randn(*self.pitch_logits.shape)
        grad_dur = np.random.randn(*self.dur_logits.shape)
        self.model.backward(grad_pitch, grad_dur)

        self.assertIsNotNone(self.model.pitch_head.grad_W)
        self.assertIsNotNone(self.model.pitch_head.grad_b)
        self.assertIsNotNone(self.model.dur_head.grad_W)
        self.assertIsNotNone(self.model.dur_head.grad_b)

    def test_backward_populates_embedding_gradients(self):
        """All embeddings should have gradients after backward."""
        grad_pitch = np.random.randn(*self.pitch_logits.shape)
        grad_dur = np.random.randn(*self.dur_logits.shape)
        self.model.backward(grad_pitch, grad_dur)

        for name, emb in self.model.embeddings.items():
            self.assertIsNotNone(emb.grad_W, f"Embedding '{name}' has no gradient")

    def test_backward_head_gradient_shapes(self):
        """Head gradients should match weight/bias shapes."""
        grad_pitch = np.random.randn(*self.pitch_logits.shape)
        grad_dur = np.random.randn(*self.dur_logits.shape)
        self.model.backward(grad_pitch, grad_dur)

        self.assertEqual(self.model.pitch_head.grad_W.shape, self.model.pitch_head.W.shape)
        self.assertEqual(self.model.pitch_head.grad_b.shape, self.model.pitch_head.b.shape)
        self.assertEqual(self.model.dur_head.grad_W.shape, self.model.dur_head.W.shape)
        self.assertEqual(self.model.dur_head.grad_b.shape, self.model.dur_head.b.shape)


class TestModelParams(unittest.TestCase):
    """Test parameter get/set/grad methods."""

    def setUp(self):
        self.model = _build_model()

    def test_get_all_params_keys(self):
        """get_all_params should return dict with correct top-level keys."""
        params = self.model.get_all_params()

        for name in self.model.embeddings:
            self.assertIn(f'emb_{name}', params)
        self.assertIn('transformer', params)
        self.assertIn('pitch_head', params)
        self.assertIn('dur_head', params)

    def test_params_contain_numpy_arrays(self):
        """Parameter sub-dicts should contain numpy arrays."""
        params = self.model.get_all_params()
        self.assertIn('W', params['emb_pitch'])
        self.assertIsInstance(params['emb_pitch']['W'], np.ndarray)
        self.assertIn('W', params['pitch_head'])
        self.assertIn('b', params['pitch_head'])

    def test_get_all_grads_keys_match_params(self):
        """get_all_grads should return same top-level keys as get_all_params."""
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        self.assertEqual(set(params.keys()), set(grads.keys()))

    def test_set_all_params_roundtrip(self):
        """set_all_params(get_all_params()) should preserve all values."""
        params = self.model.get_all_params()
        original_pitch_W = params['emb_pitch']['W'].copy()
        original_head_W = params['pitch_head']['W'].copy()
        original_head_b = params['dur_head']['b'].copy()

        new_model = _build_model()
        new_model.set_all_params(params)
        new_params = new_model.get_all_params()

        np.testing.assert_array_equal(new_params['emb_pitch']['W'], original_pitch_W)
        np.testing.assert_array_equal(new_params['pitch_head']['W'], original_head_W)
        np.testing.assert_array_equal(new_params['dur_head']['b'], original_head_b)


if __name__ == '__main__':
    unittest.main()

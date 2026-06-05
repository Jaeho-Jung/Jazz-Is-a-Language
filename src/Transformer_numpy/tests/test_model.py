"""
test_model.py
Unit tests for JazzTransformer (7-feature, GPT-style).
Mocks the Transformer backbone to test model wiring in isolation.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.layers.embedding import Embedding
from src.Transformer_numpy.layers.linear import Linear

# ---------------------------------------------------------------------------
# Constants matching config.py and model.py
# ---------------------------------------------------------------------------
VOCAB_SIZE_PITCH       = 129
VOCAB_SIZE_REL_PITCH   = 13
VOCAB_SIZE_GRID_POS    = 48
VOCAB_SIZE_CHORD_ROOT  = 13
VOCAB_SIZE_CHORD_QUALITY = 7
VOCAB_SIZE_PREV_INTERVAL = 25
NUM_DUR_CLASSES        = 16

EMBED_SIZE_PITCH         = 32
EMBED_SIZE_REL_PITCH     = 8
EMBED_SIZE_DURATION      = 8
EMBED_SIZE_GRID_POS      = 8
EMBED_SIZE_CHORD_ROOT    = 8
EMBED_SIZE_CHORD_QUALITY = 4
EMBED_SIZE_PREV_INTERVAL = 8

TOTAL_EMBED_SIZE = 76   # 32+8+8+8+8+4+8
TRANSFORMER_HIDDEN_SIZE = 128

EMBEDDING_SPEC = {
    'pitch':         (VOCAB_SIZE_PITCH,         EMBED_SIZE_PITCH),
    'rel_pitch':     (VOCAB_SIZE_REL_PITCH,      EMBED_SIZE_REL_PITCH),
    'duration':      (NUM_DUR_CLASSES,           EMBED_SIZE_DURATION),
    'prev_interval': (VOCAB_SIZE_PREV_INTERVAL,  EMBED_SIZE_PREV_INTERVAL),
    'chord_root':    (VOCAB_SIZE_CHORD_ROOT,     EMBED_SIZE_CHORD_ROOT),
    'chord_quality': (VOCAB_SIZE_CHORD_QUALITY,  EMBED_SIZE_CHORD_QUALITY),
    'metric_pos':    (VOCAB_SIZE_GRID_POS,       EMBED_SIZE_GRID_POS),
}


def _build_model():
    """Build a JazzTransformer with a mocked Transformer backbone."""
    model = type('JazzTransformer', (), {})()

    model.embeddings = {
        name: Embedding(vocab_size, embed_dim)
        for name, (vocab_size, embed_dim) in EMBEDDING_SPEC.items()
    }
    model.transformer = MagicMock()
    model.pitch_head = Linear(TRANSFORMER_HIDDEN_SIZE, VOCAB_SIZE_PITCH)
    model.dur_head   = Linear(TRANSFORMER_HIDDEN_SIZE, NUM_DUR_CLASSES)
    model.num_dur_classes = NUM_DUR_CLASSES
    model._last_features  = {}

    from src.Transformer_numpy.model import JazzTransformer as _JT
    model.forward       = _JT.forward.__get__(model, type(model))
    model.backward      = _JT.backward.__get__(model, type(model))
    model.get_all_params = _JT.get_all_params.__get__(model, type(model))
    model.get_all_grads  = _JT.get_all_grads.__get__(model, type(model))
    model.set_all_params = _JT.set_all_params.__get__(model, type(model))

    return model


def _make_features(batch_size, seq_len):
    return {
        name: np.random.randint(0, vocab_size, (batch_size, seq_len))
        for name, (vocab_size, _) in EMBEDDING_SPEC.items()
    }


class TestModelInit(unittest.TestCase):

    def setUp(self):
        self.model = _build_model()

    def test_all_embedding_keys_present(self):
        for key in EMBEDDING_SPEC:
            self.assertIn(key, self.model.embeddings, f"Missing embedding: {key}")

    def test_embedding_dimensions(self):
        for name, (vocab_size, embed_dim) in EMBEDDING_SPEC.items():
            emb = self.model.embeddings[name]
            self.assertEqual(emb.num_embeddings, vocab_size, f"{name} vocab size mismatch")
            self.assertEqual(emb.embedding_dim,  embed_dim,  f"{name} embed dim mismatch")

    def test_output_head_dimensions(self):
        self.assertEqual(self.model.pitch_head.input_features,  TRANSFORMER_HIDDEN_SIZE)
        self.assertEqual(self.model.pitch_head.output_features, VOCAB_SIZE_PITCH)
        self.assertEqual(self.model.dur_head.input_features,    TRANSFORMER_HIDDEN_SIZE)
        self.assertEqual(self.model.dur_head.output_features,   NUM_DUR_CLASSES)

    def test_exactly_seven_embeddings(self):
        self.assertEqual(len(self.model.embeddings), 7)


class TestModelForward(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len    = 5
        self.model      = _build_model()
        self.features   = _make_features(self.batch_size, self.seq_len)

        self.transformer_output = np.random.randn(
            self.batch_size, self.seq_len, TRANSFORMER_HIDDEN_SIZE)
        self.model.transformer.forward.return_value = self.transformer_output

    def test_output_shapes_gpt_style(self):
        """Forward should return (B, T, vocab) for all positions."""
        pitch_logits, dur_logits = self.model.forward(self.features)
        self.assertEqual(pitch_logits.shape, (self.batch_size, self.seq_len, VOCAB_SIZE_PITCH))
        self.assertEqual(dur_logits.shape,   (self.batch_size, self.seq_len, NUM_DUR_CLASSES))

    def test_transformer_receives_concatenated_embeddings(self):
        """Transformer input should be (B, T, TOTAL_EMBED_SIZE)."""
        self.model.forward(self.features)
        call_args = self.model.transformer.forward.call_args
        input_to_transformer = call_args[0][0]
        self.assertEqual(input_to_transformer.shape,
                         (self.batch_size, self.seq_len, TOTAL_EMBED_SIZE))

    def test_stores_last_features(self):
        self.model.forward(self.features)
        self.assertEqual(set(self.model._last_features.keys()), set(self.features.keys()))


class TestModelBackward(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len    = 5
        self.model      = _build_model()
        self.features   = _make_features(self.batch_size, self.seq_len)

        transformer_output = np.random.randn(self.batch_size, self.seq_len, TRANSFORMER_HIDDEN_SIZE)
        self.model.transformer.forward.return_value = transformer_output
        self.model.transformer.backward.return_value = np.random.randn(
            self.batch_size, self.seq_len, TOTAL_EMBED_SIZE)

        self.pitch_logits, self.dur_logits = self.model.forward(self.features)

    def test_backward_completes(self):
        grad_pitch = np.random.randn(*self.pitch_logits.shape)
        grad_dur   = np.random.randn(*self.dur_logits.shape)
        self.model.backward(grad_pitch, grad_dur)
        self.model.transformer.backward.assert_called_once()

    def test_backward_populates_head_gradients(self):
        self.model.backward(
            np.random.randn(*self.pitch_logits.shape),
            np.random.randn(*self.dur_logits.shape))
        self.assertIsNotNone(self.model.pitch_head.grad_W)
        self.assertIsNotNone(self.model.pitch_head.grad_b)
        self.assertIsNotNone(self.model.dur_head.grad_W)
        self.assertIsNotNone(self.model.dur_head.grad_b)

    def test_backward_populates_embedding_gradients(self):
        self.model.backward(
            np.random.randn(*self.pitch_logits.shape),
            np.random.randn(*self.dur_logits.shape))
        for name, emb in self.model.embeddings.items():
            self.assertIsNotNone(emb.grad_W, f"Embedding '{name}' has no gradient")

    def test_head_gradient_shapes(self):
        self.model.backward(
            np.random.randn(*self.pitch_logits.shape),
            np.random.randn(*self.dur_logits.shape))
        self.assertEqual(self.model.pitch_head.grad_W.shape, self.model.pitch_head.W.shape)
        self.assertEqual(self.model.pitch_head.grad_b.shape, self.model.pitch_head.b.shape)


class TestModelParams(unittest.TestCase):

    def setUp(self):
        self.model = _build_model()

    def test_get_all_params_keys(self):
        params = self.model.get_all_params()
        for name in self.model.embeddings:
            self.assertIn(f'emb_{name}', params)
        self.assertIn('transformer', params)
        self.assertIn('pitch_head', params)
        self.assertIn('dur_head', params)

    def test_params_contain_numpy_arrays(self):
        params = self.model.get_all_params()
        self.assertIsInstance(params['emb_pitch']['W'], np.ndarray)
        self.assertIn('W', params['pitch_head'])
        self.assertIn('b', params['pitch_head'])

    def test_set_all_params_roundtrip(self):
        params = self.model.get_all_params()
        original_pitch_W = params['emb_pitch']['W'].copy()
        original_head_W  = params['pitch_head']['W'].copy()

        new_model = _build_model()
        new_model.set_all_params(params)
        new_params = new_model.get_all_params()

        np.testing.assert_array_equal(new_params['emb_pitch']['W'], original_pitch_W)
        np.testing.assert_array_equal(new_params['pitch_head']['W'], original_head_W)


if __name__ == '__main__':
    unittest.main()

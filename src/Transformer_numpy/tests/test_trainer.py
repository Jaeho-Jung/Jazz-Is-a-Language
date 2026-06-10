import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.trainer import Trainer

B, T = 2, 4          # batch, seq_len
V_PITCH, V_DUR = 21, 10


def _make_batch():
    features = {
        'pitch':         np.zeros((B, T), dtype=np.int64),
        'rel_pitch':     np.zeros((B, T), dtype=np.int64),
        'duration':      np.zeros((B, T), dtype=np.int64),
        'prev_interval': np.zeros((B, T), dtype=np.int64),
        'chord_root':    np.zeros((B, T), dtype=np.int64),
        'chord_quality': np.zeros((B, T), dtype=np.int64),
        'metric_pos':    np.zeros((B, T), dtype=np.int64),
    }
    targets = {
        'pitch':    np.zeros((B, T), dtype=np.int64),
        'duration': np.zeros((B, T), dtype=np.int64),
    }
    return features, targets


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.mock_model     = MagicMock()
        self.mock_optimizer = MagicMock()

        # GPT-style: batch = (features_dict, targets_dict)
        self.mock_batch       = _make_batch()
        self.mock_train_loader = [self.mock_batch]
        self.mock_val_loader   = [self.mock_batch]

        self.trainer = Trainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
        )

    def test_initialization(self):
        self.assertEqual(self.trainer.epoch, 0)
        self.assertEqual(self.trainer.best_val_loss, float('inf'))

    def test_train_epoch(self):
        pitch_logits = np.random.randn(B, T, V_PITCH)
        dur_logits   = np.random.randn(B, T, V_DUR)
        self.mock_model.forward.return_value = (pitch_logits, dur_logits)
        self.mock_model.get_all_grads.return_value = {}

        avg_loss = self.trainer.train_epoch()

        self.mock_model.forward.assert_called_once()
        self.mock_model.backward.assert_called_once()
        self.trainer.optimizer.step.assert_called_once()
        self.assertEqual(self.trainer.epoch, 1)
        self.assertIsInstance(avg_loss, float)

    def test_validate(self):
        # All predictions correct for pitch, all wrong for duration
        pitch_logits = np.zeros((B, T, V_PITCH))
        dur_logits   = np.zeros((B, T, V_DUR))
        pitch_logits[:, :, 0] = 100  # target is 0 everywhere → all correct
        dur_logits[:, :, 5]   = 100  # target is 0, pred is 5 → all wrong

        self.mock_model.forward.return_value = (pitch_logits, dur_logits)

        val_loss, pitch_acc, dur_acc = self.trainer.validate()

        self.assertIsInstance(val_loss, float)
        self.assertAlmostEqual(pitch_acc, 1.0)
        self.assertAlmostEqual(dur_acc,   0.0)

    def test_convert_batch(self):
        features, target_pitch, target_dur = self.trainer._convert_batch(self.mock_batch)
        self.assertIn('pitch', features)
        self.assertEqual(target_pitch.shape, (B, T))
        self.assertEqual(target_dur.shape,   (B, T))

    def test_compute_loss_returns_scalars(self):
        pitch_logits = np.random.randn(B, T, V_PITCH)
        dur_logits   = np.random.randn(B, T, V_DUR)
        targets_p    = np.zeros((B, T), dtype=np.int64)
        targets_d    = np.zeros((B, T), dtype=np.int64)

        loss, p_loss, d_loss = self.trainer._compute_loss(
            pitch_logits, dur_logits, targets_p, targets_d)
        self.assertIsInstance(loss,   float)
        self.assertIsInstance(p_loss, float)

    def test_compute_gradients_shapes(self):
        pitch_logits = np.random.randn(B, T, V_PITCH)
        dur_logits   = np.random.randn(B, T, V_DUR)
        targets_p    = np.zeros((B, T), dtype=np.int64)
        targets_d    = np.zeros((B, T), dtype=np.int64)

        gp, gd = self.trainer._compute_gradients(
            pitch_logits, dur_logits, targets_p, targets_d)
        self.assertEqual(gp.shape, (B, T, V_PITCH))
        self.assertEqual(gd.shape, (B, T, V_DUR))


if __name__ == '__main__':
    unittest.main()

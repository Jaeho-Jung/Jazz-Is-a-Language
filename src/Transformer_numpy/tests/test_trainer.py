
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_optimizer = MagicMock()
        
        # Mock dataset/dataloader
        # batch structure: (features_dict, target_pitch, target_dur)
        self.mock_features = {
            'rel_pitch': np.array([[1, 2], [3, 4]]),
            'dur': np.array([[1, 1], [2, 2]])
        }
        self.mock_target_pitch = np.array([10, 20])
        self.mock_target_dur = np.array([1, 2])
        
        self.mock_batch = (
            self.mock_features,
            self.mock_target_pitch, 
            self.mock_target_dur
        )
        
        self.mock_train_loader = [self.mock_batch]
        self.mock_val_loader = [self.mock_batch]
        
        self.trainer = Trainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader
        )

    def test_initialization(self):
        self.assertEqual(self.trainer.model, self.mock_model)
        self.assertEqual(self.trainer.optimizer, self.mock_optimizer)
        self.assertEqual(self.trainer.epoch, 0)
        self.assertEqual(self.trainer.best_val_loss, float('inf'))

    @patch('src.Transformer_numpy.trainer.cross_entropy_loss')
    @patch('src.Transformer_numpy.trainer.cross_entropy_loss_grad')
    def test_train_epoch(self, mock_loss_grad, mock_loss):
        # Mock model output
        vocab_size = 10
        batch_size = 2
        logits_pitch = np.random.randn(batch_size, vocab_size)
        logits_dur = np.random.randn(batch_size, vocab_size)
        
        # Ensure logits are large enough for target indices (10, 20)
        # targets in setUp are 10, 20. Vocab size must be > 20.
        vocab_size = 21
        logits_pitch = np.random.randn(batch_size, vocab_size)
        logits_dur = np.random.randn(batch_size, vocab_size)
        
        self.mock_model.forward.return_value = (logits_pitch, logits_dur)
        
        # Mock loss return
        mock_loss.return_value = 0.5
        
        # Mock gradients: return shape of logits
        mock_loss_grad.return_value = np.zeros_like(logits_pitch) 
        
        # Run train epoch
        # This will call real to_one_hot and softmax from utils due to no patch
        avg_loss = self.trainer.train_epoch()
        
        # Verify calls
        self.mock_model.forward.assert_called()
        self.mock_model.backward.assert_called()
        self.trainer.optimizer.step.assert_called()

    @patch('src.Transformer_numpy.trainer.to_numpy')
    @patch('src.Transformer_numpy.trainer.cross_entropy_loss')
    @patch('src.Transformer_numpy.trainer.softmax')
    def test_validate(self, mock_softmax, mock_loss, mock_to_numpy):
        mock_to_numpy.side_effect = lambda x: x
        
        mock_loss.return_value = 0.5
        
        # Mock model call (validate uses __call__ or forward?)
        # Trainer uses self.model(features) which implies __call__
        # We need to mock __call__ on the mock_model
        
        vocab_size = 128 # default pitch vocab
        batch_size = 2
        # Create logits that ensure specific predictions for accuracy test
        # target pitch [10, 20]
        logits_pitch = np.zeros((batch_size, vocab_size))
        logits_pitch[0, 10] = 100 # Correct
        logits_pitch[1, 20] = 100 # Correct
        
        # target dur [1, 2]
        logits_dur = np.zeros((batch_size, vocab_size))
        logits_dur[0, 1] = 100 # Correct
        logits_dur[1, 5] = 100 # Incorrect (target is 2) (wait target mock is [1, 2])
        
        self.mock_model.side_effect = lambda x: (logits_pitch, logits_dur)

        val_loss, pitch_acc, dur_acc = self.trainer.validate()
        
        self.assertEqual(val_loss, 1.0) # 0.5 + 0.5
        self.assertEqual(pitch_acc, 1.0) # Both correct
        self.assertEqual(dur_acc, 0.5) # One correct

if __name__ == '__main__':
    unittest.main()

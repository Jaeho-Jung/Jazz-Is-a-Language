import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Transformer_numpy.dataset import JazzDataset

class TestJazzDataset(unittest.TestCase):
    def setUp(self):
        # Create dummy data matching dataset.py expectations
        data = {
            'melid': [1, 1, 1, 1, 1, 2, 2], # 5 events for id 1, 2 for id 2
            'pitch': [60, 62, 64, 65, 67, 70, 72],
            'dur_grid': [4, 4, 4, 4, 4, 8, 8],
            'pos_grid': [0, 4, 8, 12, 16, 0, 8],
            'chord_rel_pitch': [0, 2, 4, 5, 7, 0, 2],
            'chord_root': [0, 0, 0, 0, 5, 7, 7],
            'chord_root_rel': [0, 0, 0, 0, 5, 7, 7],
            'chord_quality': [1, 1, 1, 1, 1, 1, 1],
            'next_chord_root': [0, 0, 5, 5, 0, 0, 0],
            'next_chord_root_rel': [0, 0, 5, 5, 0, 0, 0],
            'next_chord_quality': [1, 1, 1, 1, 1, 1, 1],
        }
        self.df = pd.DataFrame(data)
        self.seq_len = 3

    @patch('pandas.read_pickle')
    def test_initialization_and_len(self, mock_read_pickle):
        """Test dataset initialization and sequence count."""
        mock_read_pickle.return_value = self.df
        dataset = JazzDataset(seq_len=self.seq_len, data_path="dummy_path")
        
        # melid 1: 5 events. seq_len=3.
        # stride = 3 // 2 = 1.
        # len_group = 5.
        # start_idx range(0, 5 - 3, 1) -> range(0, 2, 1) -> 0, 1.
        # So 2 sequences.
        
        # melid 2: 2 events. seq_len=3. sequences = 0 (skipped).
        # Total sequences = 2
        
        self.assertEqual(len(dataset), 2)
        
        # Check sequence indices (flat_offset + start_idx)
        # First seq: melid 1, start 0 -> offset 0
        # Second seq: melid 1, start 1 -> offset 1
        self.assertEqual(dataset.seq_starts[0], 0)
        self.assertEqual(dataset.seq_starts[1], 1)

    @patch('pandas.read_pickle')
    def test_getitem_shapes_and_types(self, mock_read_pickle):
        """Test the shape and type of returned items."""
        mock_read_pickle.return_value = self.df
        dataset = JazzDataset(seq_len=self.seq_len, data_path="dummy_path")
        
        # Test first item
        features, targets = dataset[0]
        
        # Check types
        self.assertIsInstance(features, dict)
        self.assertIsInstance(targets, dict)
        
        # Check shapes
        for key, val in features.items():
            self.assertEqual(val.shape, (self.seq_len,), f"Feature {key} shape mismatch")
            self.assertEqual(val.dtype, torch.long, f"Feature {key} dtype mismatch")

        # scalar targets check? No, targets are sequences too in dataset.py
        # dataset.py:
        # targets = {
        #     'pitch': torch.LongTensor(self.pitch[start+1:end+1].copy()),
        #     'duration': torch.LongTensor(self.dur[start+1:end+1].copy()),
        # }
        # So target is also seq_len long
        
        self.assertEqual(targets['pitch'].shape, (self.seq_len,))
        self.assertEqual(targets['duration'].shape, (self.seq_len,))

    @patch('pandas.read_pickle')
    def test_getitem_values(self, mock_read_pickle):
        """Test values of returned items."""
        mock_read_pickle.return_value = self.df
        dataset = JazzDataset(seq_len=self.seq_len, data_path="dummy_path")
        
        # Check values for first sequence (indices 0, 1, 2)
        # pitch: 60, 62, 64
        # target pitch (shifted by 1): 62, 64, 65
        
        features, targets = dataset[0]
        
        self.assertTrue(torch.equal(features['rel_pitch'], torch.LongTensor([0, 2, 4])))
        
        # Target
        self.assertTrue(torch.equal(targets['pitch'], torch.LongTensor([62, 64, 65])))
        self.assertTrue(torch.equal(targets['duration'], torch.LongTensor([4, 4, 4])))

    @patch('pandas.read_pickle')
    def test_nan_handling(self, mock_read_pickle):
        """Test NaNs are filled with default values."""
        # Create data with NaNs to test fillna
        df_nan = self.df.copy()
        # Use .iloc to set by integer position (label-based .loc works too since index=0..N)
        df_nan.iloc[0, df_nan.columns.get_loc('pitch')] = np.nan # Should fill with 128
        df_nan.iloc[0, df_nan.columns.get_loc('next_chord_root')] = np.nan # Should fill with 12
        
        mock_read_pickle.return_value = df_nan
        dataset = JazzDataset(seq_len=self.seq_len, data_path="dummy_path")
        
        # Just check raw arrays inside precompute logic or via getitem
        # getitem[0] covers index 0
        
        # Note: dataset.py logic:
        # self.pitch = np.concatenate(all_pitch)
        # pitch = group['pitch'].fillna(128).astype(int).values.copy()
        
        # But wait, to check pitch at index 0 via getitem target, 
        # target['pitch'] is pitch[start+1:end+1].
        # So target['pitch'][0] corresponds to original pitch[1].
        # We need to check if the input features had the filled value, but 'pitch' itself isn't a feature!
        
        # 'pitch' is only a target.
        # But wait, is pitch used as feature? No.
        # features: 'rel_pitch', 'dur', 'pos', 'chord_root', ...
        
        # So how can we test pitch fillna?
        # We can check internal attribute self.pitch[0] if it's public. Yes it is.
        self.assertEqual(dataset.pitch[0], 128)
        
        # next_chord_root IS a feature.
        features, _ = dataset[0]
        self.assertEqual(features['next_chord_root'][0].item(), 12)

    @patch('pandas.read_pickle')
    def test_target_nan_handling(self, mock_read_pickle):
        """Test target pitch NaN handling."""
        df_nan = self.df.copy()
        # Make target pitch NaN (index 3 is target for end of first sequence 0..2 if we looked at that)
        # Let's just set a value we will look at.
        # If we set index 1 to NaN.
        # input seq 0: start=0, end=3.
        # target pitch: pitch[1:4]. So includes index 1, 2, 3.
        
        df_nan.iloc[1, df_nan.columns.get_loc('pitch')] = np.nan
        
        mock_read_pickle.return_value = df_nan
        dataset = JazzDataset(seq_len=self.seq_len, data_path="dummy_path")
        
        _, targets = dataset[0] # targets['pitch'] is pitch[1], pitch[2], pitch[3]
        
        # pitch[1] should be 128
        self.assertEqual(targets['pitch'][0].item(), 128)

if __name__ == '__main__':
    unittest.main()

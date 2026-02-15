"""
Dataset for JazzTransformer (Decoder-Only) model.

Returns unified 7-feature format matching RNN/LSTM interface.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.Transformer_pytorch.config import SEQ_LEN, DATA_PATH


class JazzDataset(Dataset):
    def __init__(self, seq_len=SEQ_LEN, data_path=DATA_PATH, melids=None):
        self.seq_len = seq_len
        self.df = pd.read_pickle(data_path)

        # Build Duration Vocabulary (from FULL dataset to keep consistent mapping)
        unique_durations = sorted(self.df['dur_grid'].dropna().unique().astype(int))
        self.dur_to_idx = {dur: idx for idx, dur in enumerate(unique_durations)}
        self.idx_to_dur = {idx: dur for idx, dur in enumerate(unique_durations)}
        self.vocab_size_duration = len(unique_durations)

        # Filter by melids if provided (solo-level split to prevent data leakage)
        self.melids = melids

        # Prepare sequences
        self.sequences = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        # Group by melid to avoid crossing solo boundaries
        grouped = self.df.groupby('melid')

        self.solo_data = {}

        for melid, group in grouped:
            # Skip melids not in the provided set
            if self.melids is not None and melid not in self.melids:
                continue
            
            group = group.reset_index(drop=True)
            self.solo_data[melid] = group

            num_events = len(group)
            if num_events > self.seq_len:
                for start_idx in range(num_events - self.seq_len):
                    self.sequences.append((melid, start_idx))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        melid, start_idx = self.sequences[idx]
        solo_df = self.solo_data[melid]

        # Get window input and target 
        window_df = solo_df.iloc[start_idx : start_idx + self.seq_len]
        target_row = solo_df.iloc[start_idx + self.seq_len]

        # =================================================================
        # UNIFIED 7 FEATURES (same as RNN/LSTM)
        # =================================================================
        pitch = window_df['pitch'].fillna(128).astype(int).values
        rel_pitch = window_df['chord_rel_pitch'].fillna(12).astype(int).values
        dur = window_df['dur_grid'].apply(lambda x: self.dur_to_idx.get(x, 0)).values
        
        prev_int = window_df['prev_interval'].fillna(0).values
        prev_int_idx = np.clip(prev_int + 12, 0, 24).astype(int)
        
        chord_root = window_df['chord_root'].fillna(12).astype(int).values
        chord_quality = window_df['chord_quality'].fillna(6).astype(int).values
        metric_pos = window_df['pos_grid'].fillna(0).astype(int).values

        features = {
            'pitch': torch.LongTensor(pitch),
            'rel_pitch': torch.LongTensor(rel_pitch),
            'duration': torch.LongTensor(dur),
            'prev_interval': torch.LongTensor(prev_int_idx),
            'chord_root': torch.LongTensor(chord_root),
            'chord_quality': torch.LongTensor(chord_quality),
            'metric_pos': torch.LongTensor(metric_pos),
        }

        # =================================================================
        # TARGETS
        # =================================================================
        t_pitch_val = target_row['pitch']
        target_pitch = 128 if pd.isna(t_pitch_val) else int(t_pitch_val)
            
        t_dur_val = target_row['dur_grid']
        target_duration = self.dur_to_idx.get(t_dur_val, 0)
        
        targets = {
            'pitch': torch.tensor(target_pitch, dtype=torch.long),
            'duration': torch.tensor(target_duration, dtype=torch.long),
        }

        return features, targets
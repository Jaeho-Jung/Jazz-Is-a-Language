"""
Dataset for JazzTransformer (Decoder-Only) model.

Returns unified 7-feature format matching RNN/LSTM interface.
All data is preloaded into memory as numpy arrays for fast GPU training.
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

        # Precompute all data as contiguous numpy arrays for fast __getitem__
        self._precompute()

    def _precompute(self):
        """Precompute all features as numpy arrays — eliminates DataFrame overhead in __getitem__."""
        grouped = self.df.groupby('melid')
        
        # Collect all sequences as indices into flat arrays
        all_pitch = []
        all_rel_pitch = []
        all_dur = []
        all_prev_int = []
        all_chord_root = []
        all_chord_quality = []
        all_metric_pos = []
        
        sequences = []  # (offset_in_flat, length) for each solo
        flat_offset = 0
        
        for melid, group in grouped:
            if self.melids is not None and melid not in self.melids:
                continue
            
            group = group.reset_index(drop=True)
            n = len(group)
            
            if n <= self.seq_len:
                continue
            
            # Convert entire solo to numpy arrays once
            pitch = group['pitch'].fillna(128).astype(np.int64).values.copy()
            rel_pitch = group['chord_rel_pitch'].fillna(12).astype(np.int64).values.copy()
            dur = group['dur_grid'].apply(lambda x: self.dur_to_idx.get(x, 0)).values.astype(np.int64).copy()
            prev_int = np.clip(group['prev_interval'].fillna(0).values + 12, 0, 24).astype(np.int64).copy()
            chord_root = group['chord_root'].fillna(12).astype(np.int64).values.copy()
            chord_quality = group['chord_quality'].fillna(6).astype(np.int64).values.copy()
            metric_pos = group['pos_grid'].fillna(0).astype(np.int64).values.copy()
            
            all_pitch.append(pitch)
            all_rel_pitch.append(rel_pitch)
            all_dur.append(dur)
            all_prev_int.append(prev_int)
            all_chord_root.append(chord_root)
            all_chord_quality.append(chord_quality)
            all_metric_pos.append(metric_pos)
            
            # Record sequence indices
            for start_idx in range(n - self.seq_len):
                sequences.append((flat_offset + start_idx,))
            
            flat_offset += n
        
        # Concatenate into single contiguous arrays
        self.pitch = np.concatenate(all_pitch)
        self.rel_pitch = np.concatenate(all_rel_pitch)
        self.dur = np.concatenate(all_dur)
        self.prev_int = np.concatenate(all_prev_int)
        self.chord_root = np.concatenate(all_chord_root)
        self.chord_quality = np.concatenate(all_chord_quality)
        self.metric_pos = np.concatenate(all_metric_pos)
        
        # Store sequence start indices as numpy array
        self.seq_starts = np.array([s[0] for s in sequences], dtype=np.int64)
        
        # Free the DataFrame — no longer needed
        del self.df
        
        print(f"Precomputed {len(self.seq_starts)} sequences from {len(all_pitch)} solos "
              f"({len(self.pitch)} total events, {self.pitch.nbytes * 7 / 1024 / 1024:.0f} MB in memory)")

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx):
        start = self.seq_starts[idx]
        end = start + self.seq_len
        
        # Slice precomputed arrays (fast numpy indexing, no DataFrame overhead)
        features = {
            'pitch': torch.from_numpy(self.pitch[start:end]),
            'rel_pitch': torch.from_numpy(self.rel_pitch[start:end]),
            'duration': torch.from_numpy(self.dur[start:end]),
            'prev_interval': torch.from_numpy(self.prev_int[start:end]),
            'chord_root': torch.from_numpy(self.chord_root[start:end]),
            'chord_quality': torch.from_numpy(self.chord_quality[start:end]),
            'metric_pos': torch.from_numpy(self.metric_pos[start:end]),
        }

        targets = {
            'pitch': torch.tensor(self.pitch[end], dtype=torch.long),
            'duration': torch.tensor(self.dur[end], dtype=torch.long),
        }

        return features, targets
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.Transformer_numpy.config import DATA_PATH, SEQ_LEN

"""
relative_pitch: 0-12 (0-11: note, 12: rest)
octave: 3-6
pos: 0-47
chord_root: 0-11
chord_quality: 0-6
"""

class JazzDataset(Dataset):
    """
    Dataset for solo jazz events.
    
    Args:
        seq_len: Sequence length for input window
        data_path: Path to dataset pickle file
    """
    def __init__(self, seq_len=SEQ_LEN, data_path=DATA_PATH, melids=None):
        self.seq_len = seq_len
        self.df = pd.read_pickle(data_path)

        self.melids = melids

        # prepare sequences
        self.seq_starts = []
        self._precompute()

    def _precompute(self):
        """
        Precompute all data as numpy arrays for fast GPU training.
        """
        grouped = self.df.groupby('melid')
        
        all_pitch = []
        all_rel_pitch = []
        all_dur = []
        all_pos = []
        all_chord_root = []
        all_chord_root_rel = []
        all_chord_quality = []
        all_next_chord_root = []
        all_next_chord_root_rel = []
        all_next_chord_quality = []

        sequences = []
        flat_offset = 0

        for melid, group in grouped:
            if self.melids is not None and melid not in self.melids:
                continue

            group = group.reset_index(drop=True)
            len_group = len(group)

            if len_group <= self.seq_len:
                continue

            pitch = group['pitch'].fillna(128).astype(int).values.copy()
            rel_pitch = group['chord_rel_pitch'].fillna(12).astype(int).values.copy()
            dur = group['dur_grid'].astype(int).values.copy()
            pos = group['pos_grid'].astype(int).values.copy()
            chord_root = group['chord_root'].fillna(12).astype(int).values.copy()
            chord_root_rel = group['chord_root_rel'].fillna(12).astype(int).values.copy()
            chord_quality = group['chord_quality'].fillna(6).astype(int).values.copy()
            next_chord_root = group['next_chord_root'].fillna(12).astype(int).values.copy()
            next_chord_root_rel = group['next_chord_root_rel'].fillna(12).astype(int).values.copy()
            next_chord_quality = group['next_chord_quality'].fillna(6).astype(int).values.copy()

            all_pitch.append(pitch)
            all_rel_pitch.append(rel_pitch)
            all_dur.append(dur)
            all_pos.append(pos)
            all_chord_root.append(chord_root)
            all_chord_root_rel.append(chord_root_rel)
            all_chord_quality.append(chord_quality)
            all_next_chord_root.append(next_chord_root)
            all_next_chord_root_rel.append(next_chord_root_rel)
            all_next_chord_quality.append(next_chord_quality)

            stride = self.seq_len // 2
            for start_idx in range(0, len_group - self.seq_len, stride):
                sequences.append((flat_offset + start_idx,))

            flat_offset += len_group

        self.pitch = np.concatenate(all_pitch)
        self.rel_pitch = np.concatenate(all_rel_pitch)
        self.dur = np.concatenate(all_dur)
        self.pos = np.concatenate(all_pos)
        self.chord_root = np.concatenate(all_chord_root)
        self.chord_root_rel = np.concatenate(all_chord_root_rel)
        self.chord_quality = np.concatenate(all_chord_quality)
        self.next_chord_root = np.concatenate(all_next_chord_root)
        self.next_chord_root_rel = np.concatenate(all_next_chord_root_rel)
        self.next_chord_quality = np.concatenate(all_next_chord_quality)

        self.seq_starts = np.array([s[0] for s in sequences], dtype=np.int64)

        del self.df

        print(f"Precomputed {len(self.seq_starts)} sequences from {len(all_pitch)} solos "
              f"({len(self.pitch)} total events, {self.pitch.nbytes * 7 / 1024 / 1024:.0f} MB in memory)")

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx):
        start = self.seq_starts[idx]
        end = start + self.seq_len

        features = {
            'rel_pitch': torch.LongTensor(self.rel_pitch[start:end].copy()),
            'dur': torch.LongTensor(self.dur[start:end].copy()),
            'pos': torch.LongTensor(self.pos[start:end].copy()),
            'chord_root': torch.LongTensor(self.chord_root[start:end].copy()),
            'chord_root_rel': torch.LongTensor(self.chord_root_rel[start:end].copy()),
            'chord_quality': torch.LongTensor(self.chord_quality[start:end].copy()),
            'next_chord_root': torch.LongTensor(self.next_chord_root[start:end].copy()),
            'next_chord_root_rel': torch.LongTensor(self.next_chord_root_rel[start:end].copy()),
            'next_chord_quality': torch.LongTensor(self.next_chord_quality[start:end].copy()),
        }

        # Targets
        targets = {
            'pitch': torch.LongTensor(self.pitch[start+1:end+1].copy()),
            'duration': torch.LongTensor(self.dur[start+1:end+1].copy()),
        }

        return features, targets

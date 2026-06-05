import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.Transformer_numpy.config import DATA_PATH, SEQ_LEN


class JazzDataset(Dataset):
    """
    Dataset for JazzTransformer (NumPy, GPT-style).
    Mirrors Transformer_pytorch/dataset.py: 7 features, shifted targets.
    """

    def __init__(self, seq_len=SEQ_LEN, data_path=DATA_PATH, melids=None):
        self.seq_len = seq_len
        self.df = pd.read_pickle(data_path)

        unique_durations = sorted(self.df['dur_grid'].dropna().unique().astype(int))
        self.dur_to_idx = {dur: idx for idx, dur in enumerate(unique_durations)}
        self.idx_to_dur = {idx: dur for idx, dur in enumerate(unique_durations)}
        self.vocab_size_duration = len(unique_durations)

        self.melids = melids
        self._precompute()

    def _precompute(self):
        grouped = self.df.groupby('melid')

        all_pitch, all_rel_pitch, all_dur = [], [], []
        all_prev_int, all_chord_root, all_chord_quality, all_metric_pos = [], [], [], []

        sequences = []
        flat_offset = 0

        for melid, group in grouped:
            if self.melids is not None and melid not in self.melids:
                continue

            group = group.reset_index(drop=True)
            n = len(group)

            if n <= self.seq_len:
                continue

            pitch       = group['pitch'].fillna(128).astype(np.int64).values.copy()
            rel_pitch   = group['chord_rel_pitch'].fillna(12).astype(np.int64).values.copy()
            dur         = group['dur_grid'].apply(lambda x: self.dur_to_idx.get(x, 0)).values.astype(np.int64).copy()
            prev_int    = np.clip(group['prev_interval'].fillna(0).values + 12, 0, 24).astype(np.int64).copy()
            chord_root  = group['chord_root'].fillna(12).astype(np.int64).values.copy()
            chord_qual  = group['chord_quality'].fillna(6).astype(np.int64).values.copy()
            metric_pos  = group['pos_grid'].fillna(0).astype(np.int64).values.copy()

            all_pitch.append(pitch)
            all_rel_pitch.append(rel_pitch)
            all_dur.append(dur)
            all_prev_int.append(prev_int)
            all_chord_root.append(chord_root)
            all_chord_quality.append(chord_qual)
            all_metric_pos.append(metric_pos)

            for start_idx in range(n - self.seq_len):
                sequences.append((flat_offset + start_idx,))

            flat_offset += n

        self.pitch        = np.concatenate(all_pitch)
        self.rel_pitch    = np.concatenate(all_rel_pitch)
        self.dur          = np.concatenate(all_dur)
        self.prev_int     = np.concatenate(all_prev_int)
        self.chord_root   = np.concatenate(all_chord_root)
        self.chord_quality = np.concatenate(all_chord_quality)
        self.metric_pos   = np.concatenate(all_metric_pos)

        self.seq_starts = np.array([s[0] for s in sequences], dtype=np.int64)

        del self.df

        print(f"Precomputed {len(self.seq_starts)} sequences from {len(all_pitch)} solos "
              f"({len(self.pitch)} total events, {self.pitch.nbytes * 7 / 1024 / 1024:.0f} MB in memory)")

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx):
        start = self.seq_starts[idx]
        end   = start + self.seq_len

        features = {
            'pitch':         torch.LongTensor(self.pitch[start:end].copy()),
            'rel_pitch':     torch.LongTensor(self.rel_pitch[start:end].copy()),
            'duration':      torch.LongTensor(self.dur[start:end].copy()),
            'prev_interval': torch.LongTensor(self.prev_int[start:end].copy()),
            'chord_root':    torch.LongTensor(self.chord_root[start:end].copy()),
            'chord_quality': torch.LongTensor(self.chord_quality[start:end].copy()),
            'metric_pos':    torch.LongTensor(self.metric_pos[start:end].copy()),
        }

        targets = {
            'pitch':    torch.LongTensor(self.pitch[start+1:end+1].copy()),
            'duration': torch.LongTensor(self.dur[start+1:end+1].copy()),
        }

        return features, targets

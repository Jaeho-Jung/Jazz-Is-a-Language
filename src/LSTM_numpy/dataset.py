import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.LSTM_numpy.config import DATA_PATH, SEQ_LEN

class JazzDataset(Dataset):
    def __init__(self, seq_len=SEQ_LEN, data_path=DATA_PATH):
        self.seq_len = seq_len
        self.df = pd.read_pickle(data_path)

        # Prepare sequences
        self.sequences = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        """
        Create a list of (melid, start_idx) for valid sequences.
        Valid sequence: length seq_len + 1 (for target).
        """
        # Group by melid to avoid crossing solo boundaries
        grouped = self.df.groupby('melid')

        self.solo_data = {}

        for melid, group in grouped:
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

        # Extarct features
        pitch = window_df['pitch'].fillna(128).astype(int).values
        rel_pitch = window_df['chord_rel_pitch'].fillna(12).astype(int).values
        
        dur = window_df['dur_grid'].astype(int).values
        pos = window_df['pos_grid'].astype(int).values

        chord_root = window_df['chord_root'].fillna(12).astype(int).values
        chord_root_rel = window_df['chord_root_rel'].fillna(12).astype(int).values
        chord_quality = window_df['chord_quality'].fillna(6).astype(int).values

        next_chord_root = window_df['next_chord_root'].fillna(12).astype(int).values
        next_chord_root_rel = window_df['next_chord_root_rel'].fillna(12).astype(int).values
        next_chord_quality = window_df['next_chord_quality'].fillna(6).astype(int).values

        prev_interval = window_df['prev_interval'].fillna(0).astype(int).values
        prev_interval_idx = np.clip(prev_interval + 12, 0, 24).astype(int)

        features = {
            'pitch': torch.LongTensor(pitch),
            'rel_pitch': torch.LongTensor(rel_pitch),
            'dur': torch.LongTensor(dur),
            'pos': torch.LongTensor(pos),
            'chord_root': torch.LongTensor(chord_root),
            'chord_root_rel': torch.LongTensor(chord_root_rel),
            'chord_quality': torch.LongTensor(chord_quality),
            'next_chord_root': torch.LongTensor(next_chord_root),
            'next_chord_root_rel': torch.LongTensor(next_chord_root_rel),
            'next_chord_quality': torch.LongTensor(next_chord_quality),
            'prev_interval': torch.LongTensor(prev_interval_idx)
        }

        # Targets
        target_pitch = int(target_row['pitch']) if not pd.isna(target_row['pitch']) else 128
        target_duration = int(target_row['dur_grid'])

        return features, torch.scalar_tensor(target_pitch, dtype=torch.long), torch.scalar_tensor(target_duration, dtype=torch.long)


# if __name__ == "__main__":
#     dataset = JazzDataset()
#     print(dataset[0])
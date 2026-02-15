import sys
import os
# Add project root to sys.path to allow direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from torch.utils.data import DataLoader
from src.LSTM_pytorch import config
from src.LSTM_pytorch.dataset import JazzDataset
from src.LSTM_pytorch.model import JazzLSTM
from src.LSTM_pytorch.trainer import Trainer
import numpy as np
import pandas as pd

def main():
    print("Initializing Dataset...")
    try:
        # Perform solo-level split to prevent data leakage from key augmentation
        df = pd.read_pickle(config.DATA_PATH)
        all_melids = sorted(df['melid'].unique())
        
        np.random.seed(42)
        np.random.shuffle(all_melids)
        
        split_idx = int(0.9 * len(all_melids))
        train_melids = set(all_melids[:split_idx])
        val_melids = set(all_melids[split_idx:])
        
        print(f"Solo-level split: {len(train_melids)} train solos, {len(val_melids)} val solos")
        
        train_dataset = JazzDataset(melids=train_melids)
        val_dataset = JazzDataset(melids=val_melids)
        
        print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.DATA_PATH}")
        print("Please run preprocessing first.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Initializing Model with Duration Vocab Size: {train_dataset.vocab_size_duration}")
    model = JazzLSTM(num_dur_classes=train_dataset.vocab_size_duration)
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    print("Starting Training...")
    trainer.train()

if __name__ == "__main__":
    main()

"""
Training script for JazzTransformer.

Usage:
    python -m src.Transformer_numpy.train
"""
import torch
from torch.utils.data import DataLoader

from src.Transformer_numpy import config
from src.Transformer_numpy.dataset import JazzDataset
from src.Transformer_numpy.model import JazzTransformer
from src.Transformer_numpy.optimizer import AdamW
from src.Transformer_numpy.trainer import Trainer

import numpy as np
import pandas as pd


def main():
    # Load Dataset
    print("Initializing Dataset...")
    try:
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
        return

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = JazzTransformer(num_dur_classes=train_dataset.vocab_size_duration)
    optimizer = AdamW(model, lr=config.LEARNING_RATE)
    
    # Initialize Trainer
    trainer = Trainer(model, optimizer, train_loader, val_loader)

    # Train model
    print("Training started...")
    trainer.train()

if __name__ == "__main__":
    main()
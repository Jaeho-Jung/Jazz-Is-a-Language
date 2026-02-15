"""
Training script for JazzTransformer (Decoder-Only) model.

Usage:
    python -m src.Transformer_pytorch.train
"""

import torch
from torch.utils.data import DataLoader
from src.Transformer_pytorch import config
from src.Transformer_pytorch.dataset import JazzDataset
from src.Transformer_pytorch.model import JazzTransformer
from src.Transformer_pytorch.trainer import Trainer


import argparse
import os
import re
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Train JazzTransformer')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='models/transformer', help='Directory to save checkpoints')
    args = parser.parse_args()

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

    # Create data loaders (num_workers + pin_memory for faster GPU loading)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    model = JazzTransformer(num_dur_classes=train_dataset.vocab_size_duration)

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Determine start epoch if resuming
    start_epoch = 0
    if args.resume:
        # Try to parse epoch from filename (e.g., encoder_decoder_14.pth)
        match = re.search(r'transformer_epoch_(\d+)\.pth', args.resume)
        if match:
            start_epoch = int(match.group(1))
            print(f"Parsed start epoch {start_epoch} from checkpoint filename.")
        else:
            print("Could not parse epoch from filename. Starting from epoch 0 (but loading weights).")

    # Train model
    print(f"Training started. Checkpoints will be saved to: {args.save_dir}")
    trainer.train(save_dir=args.save_dir, resume_checkpoint=args.resume, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
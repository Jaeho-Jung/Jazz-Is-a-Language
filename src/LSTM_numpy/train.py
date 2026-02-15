"""
Training script for NumPy LSTM Jazz Solo Generator.

Usage:
    python src/LSTM_numpy/train.py
"""

import numpy as np
import sys
import os
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from torch.utils.data import DataLoader

from src.LSTM_numpy.dataset import JazzDataset
from src.LSTM_numpy.model import JazzLSTM
from src.LSTM_numpy.optimizer import Adam
from src.LSTM_numpy.trainer import Trainer
from src.LSTM_numpy import config


def get_num_dur_classes(dataset):
    """Determine number of unique duration classes from dataset."""
    all_durs = set()
    for melid, df in dataset.solo_data.items():
        all_durs.update(df['dur_grid'].dropna().astype(int).values)
    return max(all_durs) + 1


def main():
    print("=" * 60)
    print("NumPy LSTM Jazz Solo Generator - Training")
    print("=" * 60)
    
    # 1. Load Dataset
    print("\n[1/4] Loading dataset...")
    try:
        dataset = JazzDataset(
            seq_len=config.SEQ_LEN,
            data_path=config.DATA_PATH
        )
        print(f"  Total sequences: {len(dataset)}")
    except FileNotFoundError:
        print(f"  Error: Data file not found at {config.DATA_PATH}")
        print("  Please run preprocessing first.")
        return
    
    # 2. Split dataset
    print("\n[2/4] Creating train/validation split...")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"  Train: {train_size}, Validation: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    print("\n[3/4] Initializing model and optimizer...")  
    num_dur_classes = get_num_dur_classes(dataset)
    print(f"  Duration classes: {num_dur_classes}")
    
    model = JazzLSTM(num_dur_classes=num_dur_classes)
    optimizer = Adam(model, lr=config.LEARNING_RATE)
    
    # Count parameters
    total_params = 0
    for layer_name, layer_params in model.get_all_params().items():
        for param_name, param in layer_params.items():
            total_params += param.size
    print(f"  Total parameters: {total_params:,}")
    
    # 4. Train
    print("\n[4/4] Starting training...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    try:
        history = trainer.train(
            num_epochs=config.NUM_EPOCHS,
            log_interval=1,
            validate_interval=5
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        history = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses
        }
    
    # Save Model
    print("\nSaving model...")
    save_path = 'models/numpy_rnn_jazz.pkl'
    os.makedirs('models', exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'params': model.get_all_params(),
            'num_dur_classes': num_dur_classes,
            'history': history
        }, f)
    
    print(f"Model saved to: {save_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
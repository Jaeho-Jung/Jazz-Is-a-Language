"""
Trainer for JazzRNN model.

Updated to handle new (features, targets) format from dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from src.RNN_pytorch import config


class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    def _move_to_device(self, d):
        """Move dict of tensors to device."""
        return {k: v.to(self.device) for k, v in d.items()}
        
    def train(self, num_epochs=config.NUM_EPOCHS, save_dir='models/rnn'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            start_time = time.time()
            
            for i, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                train_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (i + 1) / elapsed
                    remaining_steps = len(self.train_loader) - (i + 1)
                    eta = remaining_steps / steps_per_sec
                    
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(self.train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Elapsed: {elapsed:.2f}s, ETA: {eta:.0f}s")
            
            avg_train_loss = train_loss / len(self.train_loader)
            val_loss = self.validate()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f}")
            
            torch.save(self.model.state_dict(), os.path.join(save_dir, f'rnn_epoch_{epoch+1}.pth'))
            
    def train_step(self, batch):
        features, targets = batch
        
        features = self._move_to_device(features)
        targets = self._move_to_device(targets)
        
        self.optimizer.zero_grad()
        
        # Model computes loss internally
        pitch_logits, dur_logits, loss = self.model(features, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                features, targets = batch
                features = self._move_to_device(features)
                targets = self._move_to_device(targets)
                
                _, _, loss = self.model(features, targets)
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

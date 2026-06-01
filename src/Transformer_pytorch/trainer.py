"""
Trainer for JazzTransformer (Decoder-Only) model.

Supports mixed precision training (AMP) for ~2x speedup on T4/V100/A100.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import os
import src.Transformer_pytorch.config as config



class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, num_epochs=config.NUM_EPOCHS, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Scheduler: Cosine Annealing with Warm Restarts
        # T_0: Number of iterations for the first restart
        # T_mult: A factor increases T_i after a restart
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )
        self.num_epochs = num_epochs
        
        # Mixed precision scaler (only active on CUDA)
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Mixed precision training (AMP) enabled")

    def train(self, save_dir='models/transformer', resume_checkpoint=None, start_epoch=0, patience=5):
        os.makedirs(save_dir, exist_ok=True)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
            else:
                self.model.load_state_dict(checkpoint)
        
        epochs_without_improvement = 0

        for epoch in range(start_epoch, self.num_epochs):
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

                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Elapsed: {elapsed:.2f}s, ETA: {eta:.0f}s")
            
            avg_train_loss = train_loss / len(self.train_loader)
            val_loss = self.validate()

            print(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {avg_train_loss:.4f} Val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
            }
            save_path = os.path.join(save_dir, f"transformer_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")

            # Save best model + early stopping
            if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(save_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"Saved best model: {best_path}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {self.best_val_loss:.4f}")
                    break
            
            # Learning Rate Scheduling
            if epoch < config.WARMUP_EPOCHS:
                # Linear Warmup
                lr = config.LEARNING_RATE * (epoch + 1) / config.WARMUP_EPOCHS
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print(f"Warmup Epoch [{epoch+1}/{config.WARMUP_EPOCHS}], Learning Rate: {lr:.6f}")
            else:
                # Cosine Annealing (starts after warmup)
                self.scheduler.step(epoch - config.WARMUP_EPOCHS)
                print(f"Cosine Decay, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _move_to_device(self, feature_dict):
        """Move all tensors in a dict to device."""
        return {k: v.to(self.device) for k, v in feature_dict.items()}

    def train_step(self, batch):
        features, targets = batch

        # Move data to device
        features = self._move_to_device(features)
        targets = self._move_to_device(targets)

        # Zero grad
        self.optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(enabled=self.use_amp):
            pitch_logits, dur_logits, loss = self.model(features, targets)

        # Backward pass with scaled gradients
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first for correct norm)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def validate(self):
        self.model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                features, targets = batch

                features = self._move_to_device(features)
                targets = self._move_to_device(targets)

                with autocast(enabled=self.use_amp):
                    _, _, loss = self.model(features, targets)
                val_loss += loss.item()

        return val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

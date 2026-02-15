"""
Trainer for NumPy LSTM Jazz Model.

Handles training loop, validation, and logging.
"""

import numpy as np
import time
from src.LSTM_numpy.utils import softmax, cross_entropy_loss, cross_entropy_grad
from src.LSTM_numpy import config


class Trainer:
    """
    Training orchestrator for JazzLSTM model.
    
    Handles:
        - Training loop with forward/backward/optimizer step
        - Validation evaluation
        - Loss tracking and logging
        - Early stopping (optional)
    """
    
    def __init__(self, model, optimizer, train_loader, val_loader=None):
        """
        Args:
            model: JazzLSTM model instance
            optimizer: Optimizer instance (Adam, SGD, etc.)
            train_loader: Iterable yielding (features, target_pitch, target_dur)
            val_loader: Optional validation data loader
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
    
    def _to_numpy(self, data):
        """Convert PyTorch tensors to NumPy arrays if needed."""
        if hasattr(data, 'numpy'):
            return data.numpy()
        return data
    
    def _to_one_hot(self, indices, num_classes):
        """
        Convert class indices to one-hot vectors.
        
        Args:
            indices: Array of shape (batch,) with class indices
            num_classes: Number of classes
        
        Returns:
            one_hot: Array of shape (batch, num_classes)
        """
        batch_size = len(indices)
        one_hot = np.zeros((batch_size, num_classes))
        one_hot[np.arange(batch_size), indices] = 1.0
        return one_hot
    
    def _convert_batch(self, batch):
        """
        Convert a batch from DataLoader to NumPy feature dict.
        
        Args:
            batch: Tuple of (features_dict, target_pitch, target_dur)
        
        Returns:
            features: Dict of NumPy arrays
            target_pitch: NumPy array of shape (batch,)
            target_dur: NumPy array of shape (batch,)
        """
        features_dict, target_pitch, target_dur = batch
        
        # Convert feature dict values to numpy
        features = {}
        for key, value in features_dict.items():
            features[key] = self._to_numpy(value)
        
        # Convert targets
        target_pitch = self._to_numpy(target_pitch)
        target_dur = self._to_numpy(target_dur)
        
        return features, target_pitch, target_dur
    
    def _compute_loss(self, pitch_logits, dur_logits, target_pitch, target_dur):
        """
        Compute combined cross-entropy loss for pitch and duration.
        
        Args:
            pitch_logits: Shape (batch, num_pitch_classes)
            dur_logits: Shape (batch, num_dur_classes)
            target_pitch: Shape (batch,) - class indices
            target_dur: Shape (batch,) - class indices
        
        Returns:
            total_loss: Scalar loss value
            pitch_loss: Pitch prediction loss
            dur_loss: Duration prediction loss
        """
        # Convert targets to one-hot
        target_pitch_onehot = self._to_one_hot(target_pitch, pitch_logits.shape[1])
        target_dur_onehot = self._to_one_hot(target_dur, dur_logits.shape[1])
        
        # Compute losses
        pitch_loss = cross_entropy_loss(pitch_logits, target_pitch_onehot)
        dur_loss = cross_entropy_loss(dur_logits, target_dur_onehot)
        
        total_loss = pitch_loss + dur_loss
        
        return total_loss, pitch_loss, dur_loss
    
    def _compute_gradients(self, pitch_logits, dur_logits, target_pitch, target_dur):
        """
        Compute gradients of loss w.r.t. logits.
        
        Returns:
            grad_pitch: Gradient for pitch logits
            grad_dur: Gradient for duration logits
        """
        target_pitch_onehot = self._to_one_hot(target_pitch, pitch_logits.shape[1])
        target_dur_onehot = self._to_one_hot(target_dur, dur_logits.shape[1])
        
        grad_pitch = cross_entropy_grad(pitch_logits, target_pitch_onehot)
        grad_dur = cross_entropy_grad(dur_logits, target_dur_onehot)
        
        return grad_pitch, grad_dur
    
    def train_epoch(self):
        """
        Run one epoch of training.
        
        Returns:
            avg_loss: Average loss over all batches
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Convert batch to NumPy
            features, target_pitch, target_dur = self._convert_batch(batch)
            
            # 1. Forward pass
            pitch_logits, dur_logits = self.model.forward(features)
            
            # 2. Compute loss
            loss, _, _ = self._compute_loss(pitch_logits, dur_logits, target_pitch, target_dur)
            total_loss += loss
            num_batches += 1
            
            # 3. Compute gradients
            grad_pitch, grad_dur = self._compute_gradients(
                pitch_logits, dur_logits, target_pitch, target_dur
            )
            
            # 4. Backward pass
            self.model.backward(grad_pitch, grad_dur)
            
            # 5. Optimizer step
            self.optimizer.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        self.epoch += 1
        
        return avg_loss
    
    def validate(self):
        """
        Evaluate on validation set (no gradient computation).
        
        Returns:
            avg_loss: Average validation loss
            pitch_acc: Pitch prediction accuracy
            dur_acc: Duration prediction accuracy
        """
        if self.val_loader is None:
            return None, None, None
        
        total_loss = 0.0
        total_pitch_correct = 0
        total_dur_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch in self.val_loader:
            features, target_pitch, target_dur = self._convert_batch(batch)
            
            # Forward pass only
            pitch_logits, dur_logits = self.model.forward(features)
            
            # Compute loss
            loss, _, _ = self._compute_loss(pitch_logits, dur_logits, target_pitch, target_dur)
            total_loss += loss
            num_batches += 1
            
            # Compute accuracy
            pitch_preds = np.argmax(pitch_logits, axis=1)
            dur_preds = np.argmax(dur_logits, axis=1)
            
            total_pitch_correct += np.sum(pitch_preds == target_pitch)
            total_dur_correct += np.sum(dur_preds == target_dur)
            total_samples += len(target_pitch)
        
        avg_loss = total_loss / max(num_batches, 1)
        pitch_acc = total_pitch_correct / max(total_samples, 1)
        dur_acc = total_dur_correct / max(total_samples, 1)
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, pitch_acc, dur_acc
    
    def train(self, num_epochs, log_interval=1, validate_interval=1):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            log_interval: Print training loss every N epochs
            validate_interval: Run validation every N epochs
        
        Returns:
            history: Dict with training/validation losses
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Pitch Acc':>10} | {'Dur Acc':>10} | {'Time':>8}")
        print("-" * 70)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, pitch_acc, dur_acc = None, None, None
            if self.val_loader is not None and (epoch + 1) % validate_interval == 0:
                val_loss, pitch_acc, dur_acc = self.validate()
            
            elapsed = time.time() - start_time
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                pitch_str = f"{pitch_acc:.2%}" if pitch_acc is not None else "N/A"
                dur_str = f"{dur_acc:.2%}" if dur_acc is not None else "N/A"
                print(f"{epoch+1:>6} | {train_loss:>12.4f} | {val_str:>12} | {pitch_str:>10} | {dur_str:>10} | {elapsed:>7.2f}s")
        
        print("-" * 70)
        print("Training complete.")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
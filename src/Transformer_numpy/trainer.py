from typing import Dict, List, Tuple
import numpy as np
from src.Transformer_numpy.utils import to_numpy, to_one_hot, softmax, cross_entropy_loss, cross_entropy_loss_grad
from src.Transformer_numpy import config


class Trainer:
    """
    Trainer for JazzTransformer model.
    """
    def __init__(self, model, optimizer,train_loader, val_loader):
        """
        Args:
            model: JazzTransformer instance
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
        self.best_val_loss = float('inf')
        self.best_model_params = None
        self.epoch = 0

    def train(self, num_epochs:int=config.NUM_EPOCHS, log_interval:int=1, validate_interval:int=1) -> Dict[str, List[float]]:
        """
        Full training loop

        Args:
            num_epochs: Number of epochs to train
            log_interval: Log training loss every N batches
            validate_interval: Validate on validation set every N epochs
        
        Returns:
            history: Dictionary containing training history
        """
        print(f"Start training for {num_epochs} epochs...")
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
        print("Trainingc complete.")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def train_epoch(self) -> float:
        """
        Perform one training step

        Returns:
            avg_loss: Average loss over all batches
        """
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Convert batch to Numpy
            features, target_pitch, target_dur = self._convert_batch(batch)
        
            # 1. Forward pass
            pitch_logits, dur_logits = self.model.forward(features)

            # 2. Compute loss
            loss, _, _ = self._compute_loss(pitch_logits, dur_logits, target_pitch, target_dur)
            total_loss += loss
            num_batches += 1

            # 3. Compute gradients
            grad_pitch, grad_dur = self._compute_gradients(pitch_logits, dur_logits, target_pitch, target_dur)

            # 4. Backward pass
            self.model.backward(grad_pitch, grad_dur)

            # 5. Optimizer step
            self.optimizer.step()

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        self.epoch += 1

        return avg_loss

    def validate(self) -> Tuple[float, float, float]:
        """
        Validate the model on the validation set.

        Returns:
            val_loss: Validation loss
            pitch_acc: Pitch accuracy
            dur_acc: Duration accuracy
        """
        total_loss = 0.0
        num_batches = 0
        pitch_correct = 0
        dur_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            # Convert batch to Numpy
            features, target_pitch, target_dur = self._convert_batch(batch)

            # 1. Forward pass
            pitch_logits, dur_logits = self.model(features)

            # 2. Compute loss
            loss, pitch_loss, dur_loss = self._compute_loss(pitch_logits, dur_logits, target_pitch, target_dur)
            total_loss += loss
            num_batches += 1

            # 3. Compute accuracy
            pitch_pred = np.argmax(pitch_logits, axis=1)
            dur_pred = np.argmax(dur_logits, axis=1)
            pitch_correct += np.sum(pitch_pred == target_pitch)
            dur_correct += np.sum(dur_pred == target_dur)
            total_samples += len(target_pitch)

        val_loss = total_loss / max(num_batches, 1)
        pitch_acc = pitch_correct / max(total_samples, 1)
        dur_acc = dur_correct / max(total_samples, 1)

        return val_loss, pitch_acc, dur_acc

    def _convert_batch(self, batch: Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
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

        # Convert to numpy
        features = {}
        for key, value in features_dict.items():
            features[key] = to_numpy(value)

        target_pitch = to_numpy(target_pitch)
        target_dur = to_numpy(target_dur)

        return features, target_pitch, target_dur


    def _compute_loss(self, pitch_logits: np.ndarray, dur_logits: np.ndarray, target_pitch: np.ndarray, target_dur: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute loss for a batch.

        Args:
            pitch_logits: Pitch logits of shape (batch, vocab_size)
            dur_logits: Duration logits of shape (batch, vocab_size)
            target_pitch: Target pitch of shape (batch,)
            target_dur: Target duration of shape (batch,)
        
        Returns:
            loss: Total loss
            pitch_loss: Pitch loss
            dur_loss: Duration loss
        """
        pitch_loss = cross_entropy_loss(softmax(pitch_logits), target_pitch)
        dur_loss = cross_entropy_loss(softmax(dur_logits), target_dur)
        loss = pitch_loss + dur_loss
        return loss, pitch_loss, dur_loss

    def _compute_gradients(self, pitch_logits: np.ndarray, dur_logits: np.ndarray, target_pitch: np.ndarray, target_dur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for a batch.

        Args:
            pitch_logits: Pitch logits of shape (batch, vocab_size)
            dur_logits: Duration logits of shape (batch, vocab_size)
            target_pitch: Target pitch of shape (batch,)
            target_dur: Target duration of shape (batch,)
        
        Returns:
            grad_pitch: Gradients for pitch
            grad_dur: Gradients for duration
        """
        target_pitch_onehot = to_one_hot(target_pitch, pitch_logits.shape[1])
        target_dur_onehot = to_one_hot(target_dur, dur_logits.shape[1])

        # cross_entropy_loss_grad expects probabilities, not logits
        grad_pitch = cross_entropy_loss_grad(softmax(pitch_logits), target_pitch_onehot)
        grad_dur = cross_entropy_loss_grad(softmax(dur_logits), target_dur_onehot)

        return grad_pitch, grad_dur
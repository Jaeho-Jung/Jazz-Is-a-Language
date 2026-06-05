import time
from typing import Dict, List, Tuple
import numpy as np
from src.Transformer_numpy.utils import to_numpy, to_one_hot, softmax, cross_entropy_loss, cross_entropy_loss_grad
from src.Transformer_numpy import config


def _iter_arrays(d):
    """Yield all numpy arrays from a nested dict."""
    if isinstance(d, np.ndarray):
        yield d
    elif isinstance(d, dict):
        for v in d.values():
            yield from _iter_arrays(v)


def _scale_arrays(d, scale):
    """Scale all numpy arrays in a nested dict in-place."""
    if isinstance(d, np.ndarray):
        d *= scale
    elif isinstance(d, dict):
        for v in d.values():
            _scale_arrays(v, scale)


class Trainer:
    """Trainer for JazzTransformer (NumPy, GPT-style)."""

    def __init__(self, model, optimizer, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_params = None
        self.epoch = 0

    def train(self, num_epochs: int = config.NUM_EPOCHS, log_interval: int = 1,
              validate_interval: int = 1) -> Dict[str, List[float]]:
        print(f"Start training for {num_epochs} epochs...")
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Pitch Acc':>10} | {'Dur Acc':>10} | {'Time':>8}")
        print("-" * 70)

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch()

            val_loss = pitch_acc = dur_acc = None
            if self.val_loader is not None and (epoch + 1) % validate_interval == 0:
                val_loss, pitch_acc, dur_acc = self.validate()

            elapsed = time.time() - t0

            if (epoch + 1) % log_interval == 0:
                val_str   = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                pitch_str = f"{pitch_acc:.2%}" if pitch_acc is not None else "N/A"
                dur_str   = f"{dur_acc:.2%}" if dur_acc is not None else "N/A"
                print(f"{epoch+1:>6} | {train_loss:>12.4f} | {val_str:>12} | {pitch_str:>10} | {dur_str:>10} | {elapsed:>7.2f}s")

        print("-" * 70)
        print("Training complete.")
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}

    def train_epoch(self) -> float:
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            features, target_pitch, target_dur = self._convert_batch(batch)

            pitch_logits, dur_logits = self.model.forward(features)

            loss, _, _ = self._compute_loss(pitch_logits, dur_logits, target_pitch, target_dur)
            total_loss += loss
            num_batches += 1

            grad_pitch, grad_dur = self._compute_gradients(pitch_logits, dur_logits, target_pitch, target_dur)

            self.model.backward(grad_pitch, grad_dur)

            self._clip_grad_norm(max_norm=1.0)

            self.optimizer.step()

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        self.epoch += 1
        return avg_loss

    def validate(self) -> Tuple[float, float, float]:
        total_loss = 0.0
        num_batches = 0
        pitch_correct = 0
        dur_correct = 0
        total_tokens = 0

        for batch in self.val_loader:
            features, target_pitch, target_dur = self._convert_batch(batch)

            pitch_logits, dur_logits = self.model.forward(features)

            loss, _, _ = self._compute_loss(pitch_logits, dur_logits, target_pitch, target_dur)
            total_loss += loss
            num_batches += 1

            # Accuracy over all positions
            pitch_pred = np.argmax(pitch_logits, axis=-1)  # (B, T)
            dur_pred   = np.argmax(dur_logits, axis=-1)
            pitch_correct += np.sum(pitch_pred == target_pitch)
            dur_correct   += np.sum(dur_pred == target_dur)
            total_tokens  += target_pitch.size

        val_loss  = total_loss / max(num_batches, 1)
        pitch_acc = pitch_correct / max(total_tokens, 1)
        dur_acc   = dur_correct / max(total_tokens, 1)
        return val_loss, pitch_acc, dur_acc

    def _convert_batch(self, batch):
        """Convert DataLoader batch (features_dict, targets_dict) to numpy."""
        features_dict, targets_dict = batch

        features     = {k: to_numpy(v) for k, v in features_dict.items()}
        target_pitch = to_numpy(targets_dict['pitch'])     # (B, T)
        target_dur   = to_numpy(targets_dict['duration'])  # (B, T)
        return features, target_pitch, target_dur

    def _compute_loss(self, pitch_logits, dur_logits, target_pitch, target_dur):
        """Cross-entropy over all positions (flatten B×T)."""
        B, T, V_pitch = pitch_logits.shape
        V_dur = dur_logits.shape[-1]

        pitch_probs = softmax(pitch_logits.reshape(-1, V_pitch))       # (B*T, V)
        dur_probs   = softmax(dur_logits.reshape(-1, V_dur))

        target_pitch_oh = to_one_hot(target_pitch.reshape(-1), V_pitch)
        target_dur_oh   = to_one_hot(target_dur.reshape(-1), V_dur)

        pitch_loss = cross_entropy_loss(pitch_probs, target_pitch_oh)
        dur_loss   = cross_entropy_loss(dur_probs, target_dur_oh)
        return pitch_loss + dur_loss, pitch_loss, dur_loss

    def _compute_gradients(self, pitch_logits, dur_logits, target_pitch, target_dur):
        """Compute CE gradients and reshape back to (B, T, V)."""
        B, T, V_pitch = pitch_logits.shape
        V_dur = dur_logits.shape[-1]

        pitch_probs = softmax(pitch_logits.reshape(-1, V_pitch))
        dur_probs   = softmax(dur_logits.reshape(-1, V_dur))

        target_pitch_oh = to_one_hot(target_pitch.reshape(-1), V_pitch)
        target_dur_oh   = to_one_hot(target_dur.reshape(-1), V_dur)

        grad_pitch = cross_entropy_loss_grad(pitch_probs, target_pitch_oh).reshape(B, T, V_pitch)
        grad_dur   = cross_entropy_loss_grad(dur_probs, target_dur_oh).reshape(B, T, V_dur)
        return grad_pitch, grad_dur

    def _clip_grad_norm(self, max_norm: float = 1.0):
        """Clip all parameter gradients by global L2 norm."""
        all_grads = self.model.get_all_grads()
        arrays = [g for g in _iter_arrays(all_grads) if g is not None]
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in arrays))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            _scale_arrays(all_grads, scale)

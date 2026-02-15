"""
NumPy RNN Model for Jazz Solo Generation

Integrates embeddings, RNN, and output heads into a complete model.
"""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.RNN_numpy.layers.embedding import Embedding
from src.RNN_numpy.layers.linear import Linear
from src.RNN_numpy.layers.rnn import RNN
from src.RNN_numpy import config


class JazzRNN:
    """
    Complete RNN model for jazz solo generation.
    
    Architecture:
        Input features (9 categorical) → Embeddings → Concatenate
                                                      ↓
                                                RNN Layers
                                                      ↓
                                              Last timestep
                                         ↙                    ↘
                              Pitch Head                  Duration Head
                           (129 classes)                  (dynamic classes)
    """
    
    def __init__(self, num_dur_classes):
        """
        Args:
            num_dur_classes: Number of duration classes (vocab size for durations)
        """
        # Initialize all embeddings
        # Format: Embedding(vocab_size, embed_dim)
        self.embeddings = {
            'pitch': Embedding(config.VOCAB_SIZE_PITCH, config.EMBED_SIZE_PITCH),
            'rel_pitch': Embedding(config.VOCAB_SIZE_REL_PITCH, config.EMBED_SIZE_REL_PITCH),
            'dur': Embedding(num_dur_classes, config.EMBED_SIZE_DURATION),
            'pos': Embedding(config.VOCAB_SIZE_GRID_POS, config.EMBED_SIZE_GRID_POS),
            'chord_root': Embedding(config.VOCAB_SIZE_CHORD_ROOT, config.EMBED_SIZE_CHORD_ROOT),
            'chord_root_rel': Embedding(config.VOCAB_SIZE_CHORD_ROOT_REL, config.EMBED_SIZE_CHORD_ROOT_REL),
            'chord_quality': Embedding(config.VOCAB_SIZE_CHORD_QUALITY, config.EMBED_SIZE_CHORD_QUALITY),
            'next_chord_root': Embedding(config.VOCAB_SIZE_CHORD_ROOT, config.EMBED_SIZE_CHORD_ROOT),
            'next_chord_root_rel': Embedding(config.VOCAB_SIZE_CHORD_ROOT_REL, config.EMBED_SIZE_CHORD_ROOT_REL),
            'next_chord_quality': Embedding(config.VOCAB_SIZE_CHORD_QUALITY, config.EMBED_SIZE_CHORD_QUALITY),
            'prev_interval': Embedding(config.VOCAB_SIZE_PREV_INTERVAL, config.EMBED_SIZE_PREV_INTERVAL),
        }
        
        # Initialize RNN
        self.rnn = RNN(
            input_size=config.TOTAL_EMBED_SIZE,
            hidden_size=config.RNN_HIDDEN_SIZE
        )
        
        # Initialize output heads
        self.pitch_head = Linear(config.RNN_HIDDEN_SIZE, config.VOCAB_SIZE_PITCH)
        self.dur_head = Linear(config.RNN_HIDDEN_SIZE, num_dur_classes)
        
        self.num_dur_classes = num_dur_classes
    
    def forward(self, features):
        """
        Forward pass through entire model.
        
        Args:
            features: Dict of feature arrays, each shape (batch, seq_len)
                Keys: 'pitch', 'rel_pitch', 'dur', 'pos', 'chord_root',
                      'chord_root_rel', 'chord_quality', 'next_chord_root',
                      'next_chord_root_rel', 'next_chord_quality',
                      'prev_interval'
        
        Returns:
            pitch_logits: Pitch predictions, shape (batch, VOCAB_SIZE_PITCH)
            dur_logits: Duration predictions, shape (batch, num_dur_classes)
        """
        # 1. Embed all features
        # Each embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        emb_pitch = self.embeddings['pitch'].forward(features['pitch'])
        emb_rel = self.embeddings['rel_pitch'].forward(features['rel_pitch'])
        emb_dur = self.embeddings['dur'].forward(features['dur'])
        emb_pos = self.embeddings['pos'].forward(features['pos'])
        emb_cr = self.embeddings['chord_root'].forward(features['chord_root'])
        emb_cqr = self.embeddings['chord_root_rel'].forward(features['chord_root_rel'])
        emb_cq = self.embeddings['chord_quality'].forward(features['chord_quality'])
        emb_ncr = self.embeddings['next_chord_root'].forward(features['next_chord_root'])
        emb_ncqr = self.embeddings['next_chord_root_rel'].forward(features['next_chord_root_rel'])
        emb_ncq = self.embeddings['next_chord_quality'].forward(features['next_chord_quality'])
        emb_pint = self.embeddings['prev_interval'].forward(features['prev_interval'])
        
        # 2. Concatenate embeddings
        # Result: (batch, seq_len, total_embed_size)
        rnn_input = np.concatenate([
            emb_pitch, emb_rel, emb_dur, emb_pos,
            emb_cr, emb_cqr, emb_cq,
            emb_ncr, emb_ncqr, emb_ncq,
            emb_pint
        ], axis=-1)
        
        # 3. Pass through RNN
        # h_seq: (batch, seq_len, hidden_size)
        # h_final: (batch, hidden_size)
        self._cached_seq_len = rnn_input.shape[1]  # Cache seq_len for backward
        h_seq, h_final = self.rnn.forward(rnn_input)
        
        # 4. Take last timestep
        last_step_output = h_final  # Already the last timestep
        
        # 5. Pass through output heads
        pitch_logits = self.pitch_head.forward(last_step_output)
        dur_logits = self.dur_head.forward(last_step_output)
        
        return pitch_logits, dur_logits
    
    def backward(self, grad_pitch_logits, grad_dur_logits):
        """
        Backward pass through entire network.
        
        Args:
            grad_pitch_logits: Gradient w.r.t. pitch logits, shape (batch, VOCAB_SIZE_PITCH)
            grad_dur_logits: Gradient w.r.t. duration logits, shape (batch, num_dur_classes)
        
        Returns:
            None (gradients stored in each layer)
        """
        # Backprop through output heads
        grad_from_pitch = self.pitch_head.backward(grad_pitch_logits)
        grad_from_dur = self.dur_head.backward(grad_dur_logits)
        
        # Combine gradients from both heads
        grad_last_hidden = grad_from_pitch + grad_from_dur  # (batch, hidden_size)
        
        # Expand to sequence: only last timestep has gradient
        batch_size = grad_last_hidden.shape[0]
        seq_len = self._cached_seq_len  # Use cached seq_len from forward pass
        
        # Create gradient tensor for full sequence (zeros except last timestep)
        grad_h_seq = np.zeros((batch_size, seq_len, config.RNN_HIDDEN_SIZE))
        grad_h_seq[:, -1, :] = grad_last_hidden
        
        # Backprop through RNN
        grad_rnn_input = self.rnn.backward(grad_h_seq)
        
        # Backprop through embeddings
        # Split gradients by embedding size (must match forward concatenation order)
        idx = 0
        embed_sizes = [
            config.EMBED_SIZE_PITCH,
            config.EMBED_SIZE_REL_PITCH,
            config.EMBED_SIZE_DURATION,
            config.EMBED_SIZE_GRID_POS,
            config.EMBED_SIZE_CHORD_ROOT,
            config.EMBED_SIZE_CHORD_ROOT_REL,
            config.EMBED_SIZE_CHORD_QUALITY,
            config.EMBED_SIZE_CHORD_ROOT,        # next_chord_root
            config.EMBED_SIZE_CHORD_ROOT_REL,    # next_chord_root_rel
            config.EMBED_SIZE_CHORD_QUALITY,     # next_chord_quality
            config.EMBED_SIZE_PREV_INTERVAL
        ]
        
        embed_keys = [
            'pitch', 'rel_pitch', 'dur', 'pos',
            'chord_root', 'chord_root_rel', 'chord_quality',
            'next_chord_root', 'next_chord_root_rel', 'next_chord_quality',
            'prev_interval'
        ]
        
        for key, size in zip(embed_keys, embed_sizes):
            grad_emb = grad_rnn_input[:, :, idx:idx+size]
            # Backprop through embedding (no return value, stores grad internally)
            self.embeddings[key].backward(grad_emb)
            idx += size
    
    def get_all_params(self):
        """Get all trainable parameters."""
        params = {}
        
        # Embeddings
        for name, emb in self.embeddings.items():
            params[f'emb_{name}'] = emb.get_params()
        
        # RNN
        params['rnn_cell'] = self.rnn.cell.get_params()
        
        # Output heads
        params['pitch_head'] = self.pitch_head.get_params()
        params['dur_head'] = self.dur_head.get_params()
        
        return params
    
    def get_all_grads(self):
        """Get all gradients."""
        grads = {}
        
        # Embeddings
        for name, emb in self.embeddings.items():
            grads[f'emb_{name}'] = emb.get_grads()
        
        # RNN
        grads['rnn_cell'] = self.rnn.cell.get_grads()
        
        # Output heads
        grads['pitch_head'] = self.pitch_head.get_grads()
        grads['dur_head'] = self.dur_head.get_grads()
        
        return grads
    
    def set_all_params(self, params):
        """
        Set all trainable parameters from nested dictionary.
        
        Args:
            params: Nested dict matching structure of get_all_params()
        """
        # Embeddings
        for name, emb in self.embeddings.items():
            key = f'emb_{name}'
            if key in params:
                emb.set_params(params[key])
        
        # RNN
        if 'rnn_cell' in params:
            self.rnn.cell.set_params(params['rnn_cell'])
        
        # Output heads
        if 'pitch_head' in params:
            self.pitch_head.set_params(params['pitch_head'])
        if 'dur_head' in params:
            self.dur_head.set_params(params['dur_head'])

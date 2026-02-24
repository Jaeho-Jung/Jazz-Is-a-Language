import numpy as np
from typing import Dict, Tuple

from src.Transformer_numpy import config
from src.Transformer_numpy.layers import Embedding, Linear, Transformer

class JazzTransformer:
    """
    Transformer model for jazz solo generation.
    Predicts pitch and duration of the next note given a sequence of features.
    """

    def __init__(self, num_dur_classes: int):
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

        # Initialize Transformer
        self.transformer = Transformer(
            input_size=config.TOTAL_EMBED_SIZE,
            hidden_size=config.TRANSFORMER_HIDDEN_SIZE
        )

        # Initialize output heads
        self.pitch_head = Linear(config.TRANSFORMER_HIDDEN_SIZE, config.VOCAB_SIZE_PITCH)
        self.dur_head = Linear(config.TRANSFORMER_HIDDEN_SIZE, num_dur_classes)

        self.num_dur_classes = num_dur_classes

        self._last_features: Dict[str, np.ndarray] = {}
        self._seq_len: int = 0

    def forward(self, features: dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            features: Dictionary of features with shape (batch_size, seq_len)
        Returns:
            pitch_logits: (batch_size, seq_len, vocab_size_pitch)
            dur_logits: (batch_size, seq_len, num_dur_classes)
        """
        self._last_features = features

        # Embed features
        embedded_features = []
        for feature_name, feature_values in features.items():
            embedded_features.append(self.embeddings[feature_name](feature_values))
        
        # Concatenate embeddings
        transformer_input = np.concatenate(embedded_features, axis=-1) # (batch_size, seq_len, total_embed_size)

        # Pass through Transformer
        self._seq_len = transformer_input.shape[1]
        transformer_output = self.transformer.forward(transformer_input) # (batch_size, seq_len, hidden_size)
        
        # Take last timestep
        last_step_output = transformer_output[:, -1, :] # (batch_size, hidden_size)
        
        # Pass through output heads
        pitch_logits = self.pitch_head.forward(last_step_output) # (batch_size, vocab_size_pitch)
        dur_logits = self.dur_head.forward(last_step_output) # (batch_size, num_dur_classes)
        
        return pitch_logits, dur_logits

    def backward(self, grad_pitch_logits: np.ndarray, grad_dur_logits: np.ndarray) -> None:
        """
        Backward pass through the model.
        
        Args:
            grad_pitch_logits: Gradient of pitch logits
            grad_dur_logits: Gradient of duration logits
        """
        # Backprop through output heads
        grad_from_pitch = self.pitch_head.backward(grad_pitch_logits) # (batch_size, hidden_size)
        grad_from_dur = self.dur_head.backward(grad_dur_logits) # (batch_size, hidden_size)
        grad_last_step = grad_from_pitch + grad_from_dur # (batch_size, hidden_size)
        
        batch_size, hidden_size = grad_last_step.shape
        seq_len = self._seq_len
        grad_transformer = np.zeros((batch_size, seq_len, hidden_size))
        grad_transformer[:, -1, :] = grad_last_step

        # Backprop through Transformer
        grad_transformer = self.transformer.backward(grad_transformer) # (batch_size, seq_len, total_embed_size)
        
        # Backprop through embeddings
        offset = 0
        for feature_name, feature_values in self._last_features.items():
            embed_dim = self.embeddings[feature_name].embedding_dim
            self.embeddings[feature_name].backward(grad_transformer[:, :, offset:offset+embed_dim]) # (batch_size, seq_len, embed_dim)
            offset += embed_dim

    def get_all_params(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters."""
        params = {}
        
        # Embeddings
        for name, emb in self.embeddings.items():
            params[f'emb_{name}'] = emb.get_params()
        
        # Transformer
        params['transformer'] = self.transformer.get_params()
        
        # Output heads
        params['pitch_head'] = self.pitch_head.get_params()
        params['dur_head'] = self.dur_head.get_params()
        
        return params
    
    def get_all_grads(self) -> Dict[str, np.ndarray]:
        """Get all gradients."""
        grads = {}
        
        # Embeddings
        for name, emb in self.embeddings.items():
            grads[f'emb_{name}'] = emb.get_grads()
        
        # Transformer
        grads['transformer'] = self.transformer.get_grads()
        
        # Output heads
        grads['pitch_head'] = self.pitch_head.get_grads()
        grads['dur_head'] = self.dur_head.get_grads()
        
        return grads
    
    def set_all_params(self, params: Dict[str, np.ndarray]):
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
        
        # Transformer
        if 'transformer' in params:
            self.transformer.set_params(params['transformer'])
        
        # Output heads
        if 'pitch_head' in params:
            self.pitch_head.set_params(params['pitch_head'])
        if 'dur_head' in params:
            self.dur_head.set_params(params['dur_head'])

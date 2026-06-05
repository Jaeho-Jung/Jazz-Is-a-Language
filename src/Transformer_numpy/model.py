import numpy as np
from typing import Dict, Tuple

from src.Transformer_numpy import config
from src.Transformer_numpy.layers import Embedding, Linear, Transformer


class JazzTransformer:
    """
    GPT-style decoder-only Transformer for jazz solo generation.

    7-feature input (same as Transformer_pytorch):
        pitch, rel_pitch, duration, prev_interval,
        chord_root, chord_quality, metric_pos
    Output: pitch_logits (B, T, 129) and dur_logits (B, T, num_dur_classes)
    """

    def __init__(self, num_dur_classes: int):
        self.embeddings = {
            'pitch':         Embedding(config.VOCAB_SIZE_PITCH,         config.EMBED_SIZE_PITCH),
            'rel_pitch':     Embedding(config.VOCAB_SIZE_REL_PITCH,     config.EMBED_SIZE_REL_PITCH),
            'duration':      Embedding(num_dur_classes,                  config.EMBED_SIZE_DURATION),
            'prev_interval': Embedding(config.VOCAB_SIZE_PREV_INTERVAL, config.EMBED_SIZE_PREV_INTERVAL),
            'chord_root':    Embedding(config.VOCAB_SIZE_CHORD_ROOT,    config.EMBED_SIZE_CHORD_ROOT),
            'chord_quality': Embedding(config.VOCAB_SIZE_CHORD_QUALITY, config.EMBED_SIZE_CHORD_QUALITY),
            'metric_pos':    Embedding(config.VOCAB_SIZE_GRID_POS,      config.EMBED_SIZE_GRID_POS),
        }

        self.transformer = Transformer(
            input_size=config.TOTAL_EMBED_SIZE,
            hidden_size=config.TRANSFORMER_HIDDEN_SIZE,
        )

        self.pitch_head = Linear(config.TRANSFORMER_HIDDEN_SIZE, config.VOCAB_SIZE_PITCH)
        self.dur_head   = Linear(config.TRANSFORMER_HIDDEN_SIZE, num_dur_classes)

        self.num_dur_classes = num_dur_classes
        self._last_features: Dict[str, np.ndarray] = {}

    def forward(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            features: dict of 7 arrays, each (B, T)
        Returns:
            pitch_logits: (B, T, VOCAB_SIZE_PITCH)
            dur_logits:   (B, T, num_dur_classes)
        """
        self._last_features = features

        embedded = [self.embeddings[k].forward(features[k]) for k in self.embeddings]
        transformer_input = np.concatenate(embedded, axis=-1)         # (B, T, TOTAL_EMBED_SIZE)

        transformer_output = self.transformer.forward(transformer_input)  # (B, T, hidden)

        pitch_logits = self.pitch_head.forward(transformer_output)    # (B, T, vocab_pitch)
        dur_logits   = self.dur_head.forward(transformer_output)      # (B, T, num_dur_classes)
        return pitch_logits, dur_logits

    def backward(self, grad_pitch_logits: np.ndarray, grad_dur_logits: np.ndarray) -> None:
        """
        Args:
            grad_pitch_logits: (B, T, VOCAB_SIZE_PITCH)
            grad_dur_logits:   (B, T, num_dur_classes)
        """
        grad_from_pitch  = self.pitch_head.backward(grad_pitch_logits)  # (B, T, hidden)
        grad_from_dur    = self.dur_head.backward(grad_dur_logits)       # (B, T, hidden)
        grad_transformer = grad_from_pitch + grad_from_dur

        grad_input = self.transformer.backward(grad_transformer)         # (B, T, TOTAL_EMBED_SIZE)

        offset = 0
        for name, emb in self.embeddings.items():
            d = emb.embedding_dim
            emb.backward(grad_input[:, :, offset:offset + d])
            offset += d

    def get_all_params(self) -> Dict[str, dict]:
        params = {f'emb_{k}': emb.get_params() for k, emb in self.embeddings.items()}
        params['transformer'] = self.transformer.get_params()
        params['pitch_head']  = self.pitch_head.get_params()
        params['dur_head']    = self.dur_head.get_params()
        return params

    def get_all_grads(self) -> Dict[str, dict]:
        grads = {f'emb_{k}': emb.get_grads() for k, emb in self.embeddings.items()}
        grads['transformer'] = self.transformer.get_grads()
        grads['pitch_head']  = self.pitch_head.get_grads()
        grads['dur_head']    = self.dur_head.get_grads()
        return grads

    def set_all_params(self, params: Dict[str, dict]) -> None:
        for name, emb in self.embeddings.items():
            key = f'emb_{name}'
            if key in params:
                emb.set_params(params[key])
        if 'transformer' in params:
            self.transformer.set_params(params['transformer'])
        if 'pitch_head' in params:
            self.pitch_head.set_params(params['pitch_head'])
        if 'dur_head' in params:
            self.dur_head.set_params(params['dur_head'])

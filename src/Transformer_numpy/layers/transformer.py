"""
transformer.py
NumPy decoder-only Transformer backbone.
Provides: MultiHeadCausalSelfAttention, TransformerBlock, Transformer
"""

import numpy as np
from typing import Dict, List, Optional

from src.Transformer_numpy.layers.linear import Linear
from src.Transformer_numpy.layers.layer_norm import LayerNorm
from src.Transformer_numpy.layers.feed_forward import FeedForward
from src.Transformer_numpy.layers.dropout import Dropout
from src.Transformer_numpy import config
from src.Transformer_numpy.utils import softmax


class MultiHeadCausalSelfAttention:
    """
    Multi-head causal self-attention with single Q/K/V projections.
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout_rate: float = config.DROPOUT_RATE,
                 max_seq_len: int = config.MAX_SEQ_LEN):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.attn_dropout = Dropout(dropout_rate)
        self.causal_mask = np.tril(np.ones((max_seq_len, max_seq_len)))

        self._q: Optional[np.ndarray] = None
        self._k: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._attn_weights: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape

        # Project and reshape to (B, h, T, d)
        q = self.q_proj.forward(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj.forward(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj.forward(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        self._q, self._k, self._v = q, k, v

        # Attention scores (B, h, T, T)
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)

        # Causal mask: set future positions to large negative
        mask = self.causal_mask[:T, :T]
        attn_scores = np.where(mask[np.newaxis, np.newaxis, :, :] == 1, attn_scores, -1e9)

        attn_weights = softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout.forward(attn_weights)
        self._attn_weights = attn_weights

        # (B, h, T, d) → (B, T, C)
        out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj.forward(out)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        B, T, C = grad_out.shape
        h, d = self.num_heads, self.head_dim
        q, k, v, attn = self._q, self._k, self._v, self._attn_weights

        # out_proj backward → (B, T, C)
        grad_out = self.out_proj.backward(grad_out)

        # Reshape to heads (B, h, T, d)
        grad_out_heads = grad_out.reshape(B, T, h, d).transpose(0, 2, 1, 3)

        # attn_weights @ v → grad_v and grad_attn
        grad_v = attn.transpose(0, 1, 3, 2) @ grad_out_heads     # (B, h, T, d)
        grad_attn = grad_out_heads @ v.transpose(0, 1, 3, 2)     # (B, h, T, T)

        # Dropout backward
        grad_attn = self.attn_dropout.backward(grad_attn)

        # Softmax backward: grad_scores = attn * (grad_attn - sum(grad_attn * attn))
        grad_scores = attn * (grad_attn - (grad_attn * attn).sum(axis=-1, keepdims=True))

        # Zero out masked (future) positions
        mask = self.causal_mask[:T, :T]
        grad_scores = grad_scores * mask[np.newaxis, np.newaxis, :, :]

        # Scale
        grad_scores = grad_scores * (d ** -0.5)

        # q @ k.T → grad_q and grad_k
        grad_q = grad_scores @ k                              # (B, h, T, d)
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ q       # (B, h, T, d)

        # Reshape back to (B, T, C)
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(B, T, C)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(B, T, C)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(B, T, C)

        return (self.q_proj.backward(grad_q) +
                self.k_proj.backward(grad_k) +
                self.v_proj.backward(grad_v))

    def get_params(self) -> Dict[str, dict]:
        return {
            'q_proj': self.q_proj.get_params(),
            'k_proj': self.k_proj.get_params(),
            'v_proj': self.v_proj.get_params(),
            'out_proj': self.out_proj.get_params(),
        }

    def get_grads(self) -> Dict[str, dict]:
        return {
            'q_proj': self.q_proj.get_grads(),
            'k_proj': self.k_proj.get_grads(),
            'v_proj': self.v_proj.get_grads(),
            'out_proj': self.out_proj.get_grads(),
        }

    def set_params(self, params: dict) -> None:
        self.q_proj.set_params(params['q_proj'])
        self.k_proj.set_params(params['k_proj'])
        self.v_proj.set_params(params['v_proj'])
        self.out_proj.set_params(params['out_proj'])


class TransformerBlock:
    """Pre-LN Transformer block: LN → Attn → residual → LN → FFN → residual."""

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout_rate: float = config.DROPOUT_RATE,
                 max_seq_len: int = config.MAX_SEQ_LEN):
        self.mha = MultiHeadCausalSelfAttention(embed_dim, num_heads, dropout_rate, max_seq_len)
        self.ffn = FeedForward(embed_dim, dropout_rate)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Pre-LN residual backward (both branches share the same input gradient)
        grad_x1 = grad_output + self.ln2.backward(self.ffn.backward(grad_output))
        return grad_x1 + self.ln1.backward(self.mha.backward(grad_x1))

    def get_params(self) -> Dict[str, dict]:
        return {
            'mha': self.mha.get_params(),
            'ln1': self.ln1.get_params(),
            'ln2': self.ln2.get_params(),
            'ffn': self.ffn.get_params(),
        }

    def get_grads(self) -> Dict[str, dict]:
        return {
            'mha': self.mha.get_grads(),
            'ln1': self.ln1.get_grads(),
            'ln2': self.ln2.get_grads(),
            'ffn': self.ffn.get_grads(),
        }

    def set_params(self, params: dict) -> None:
        self.mha.set_params(params['mha'])
        self.ln1.set_params(params['ln1'])
        self.ln2.set_params(params['ln2'])
        self.ffn.set_params(params['ffn'])


class Transformer:
    """
    Decoder-only Transformer backbone.
    Input: concatenated embeddings (B, T, input_size).
    Output: hidden states (B, T, hidden_size).
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_heads: int = config.NUM_HEADS,
                 num_blocks: int = config.NUM_BLOCKS,
                 dropout_rate: float = config.DROPOUT_RATE,
                 max_seq_len: int = config.MAX_SEQ_LEN):
        self.input_proj = Linear(input_size, hidden_size)
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(hidden_size, num_heads, dropout_rate, max_seq_len)
            for _ in range(num_blocks)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.input_proj.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return self.input_proj.backward(grad)

    def get_params(self) -> Dict[str, dict]:
        params = {'input_proj': self.input_proj.get_params()}
        for i, block in enumerate(self.blocks):
            params[f'block_{i}'] = block.get_params()
        return params

    def get_grads(self) -> Dict[str, dict]:
        grads = {'input_proj': self.input_proj.get_grads()}
        for i, block in enumerate(self.blocks):
            grads[f'block_{i}'] = block.get_grads()
        return grads

    def set_params(self, params: dict) -> None:
        self.input_proj.set_params(params['input_proj'])
        for i, block in enumerate(self.blocks):
            block.set_params(params[f'block_{i}'])

"""
transformer.py
NumPy decoder-only Transformer Layer
Provides: MultiHeadSelfAttention, TransformerBlock, Transformer 
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.Transformer_numpy.layers.linear import Linear
from src.Transformer_numpy.layers.embedding import Embedding
from src.Transformer_numpy.layers.layer_norm import LayerNorm

from src.Transformer_numpy.layers.feed_forward import FeedForward
from src.Transformer_numpy.layers.transformer_block import TransformerBlock
from src.Transformer_numpy.layers.dropout import Dropout
from src.Transformer_numpy.config import config
from src.Transformer_numpy.utils import softmax


class SelfAttentionHead:
    """
    Single Self-Attention Head
    """
    def __init__(self, embed_dim: int, dropout_rate: float = config.DROPOUT_RATE, max_seq_len: int = config.MAX_SEQ_LEN):
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        self.register_buffer(
            'causal_mask', np.tril(np.ones((1, max_seq_len, max_seq_len)))
        )

        self.dropout = Dropout(dropout_rate)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input, shape (batch_size, seq_len, hidden_size)
        Returns:
            y: Output, shape (batch_size, seq_len, hidden_size)
        """
        B, T, C = x.shape

        # Project and reshape: (B, T, embed_dim) -> (B, num_heads, T, head_dim)
        q = self.q_proj.forward(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj.forward(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj.forward(x).reshape(B, T, self.num_heads, self.head_dim)

        # Attention scores: (B, num_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask == 0, -np.inf)

        # Apply softmax
        attn = softmax(attn, axis=-1)
        attn = self.dropout.forward(attn)

        # Compute output
        out = attn @ v # (B, num_heads, T, head_dim)
        out = out.reshape(B, T, self.embed_dim)
        return out

class MultiHeadCausalSelfAttention:
    """
    Multi-Head Causal Self-Attention Layer
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = config.DROPOUT_RATE, max_seq_len: int = config.MAX_SEQ_LEN):
        self.heads = [SelfAttentionHead(embed_dim, dropout_rate, max_seq_len) for _ in range(num_heads)]
        self.proj = Linear(embed_dim, embed_dim)
        self.dropout = Dropout(dropout_rate)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.register_buffer(
            'causal_mask', np.tril(np.ones((1, max_seq_len, max_seq_len)))
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input, shape (batch_size, seq_len, hidden_size)
        Returns:
            y: Output, shape (batch_size, seq_len, hidden_size)
        """
        out = np.stack([head.forward(x) for head in self.heads], axis=1)
        out = self.proj.forward(out)
        out = self.dropout.forward(out)

        return out  
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Args:
            grad_output: Gradient of the output, shape (batch_size, seq_len, hidden_size)
        Returns:
            grad_input: Gradient of the input, shape (batch_size, seq_len, hidden_size)
        """
        grad_input = self.proj.backward(grad_output)
        grad_input = self.dropout.backward(grad_input)
        grad_input = np.stack([head.backward(grad_input) for head in self.heads], axis=1)
        return grad_input
        
        
        
        

class TransformerBlock:
    """
    Single decoder block using Pre-LN architecture
    
    x = x + Attn(LN(x))
    x = x + FFN(LN(x))
    """

    def __init__(self, embed_dim: int, num_heads: int):
        head_size = embed_dim // num_heads
        self.mha = MultiHeadSelfAttention(num_heads, head_size)
        self.ffn = FeedForward(embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

        self._mha_out: Optional[np.ndarray] = None
        self._ffn_out: Optional[np.ndarray] = None
        self._pre_ln1: Optional[np.ndarray] = None
        self._pre_ln2: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input, shape (batch_size, seq_len, hidden_size)
        Returns:
            y: Output, shape (batch_size, seq_len, hidden_size)
        """
        self._pre_ln1 = x
        x = self.ln1.forward(x)
        self._mha_out = self.mha.forward(x)
        x = x + self._mha_out
        self._pre_ln2 = x
        x = self.ln2.forward(x)
        self._ffn_out = self.ffn.forward(x)
        y = x + self._ffn_out
        return y
        

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Args:
            grad_output: Gradient from next layer, shape (batch_size, seq_len, hidden_size)
        Returns:
            grad_input: Gradient w.r.t. input, shape (batch_size, seq_len, hidden_size)
        """
        grad_input = grad_output
        grad_input = self.ffn.backward(grad_input)
        grad_input = grad_input + self._ffn_out
        grad_input = self.ln2.backward(grad_input)
        grad_input = grad_input + self._pre_ln2
        grad_input = self.mha.backward(grad_input)
        grad_input = grad_input + self._mha_out
        grad_input = self.ln1.backward(grad_input)
        grad_input = grad_input + self._pre_ln1
        return grad_input

    def get_params(self) -> Dict[str, dict]:
        return {
            'mha': self.mha.get_params(),
            'ln1': self.ln1.get_params(),
            'ln2': self.ln2.get_params(),
            'ffn': self.ffn.get_params()
        }

    def get_grads(self) -> Dict[str, dict]:
        return {
            'mha': self.mha.get_grads(),
            'ln1': self.ln1.get_grads(),
            'ln2': self.ln2.get_grads(),
            'ffn': self.ffn.get_grads()
        }

    def set_params(self, params: dict) -> None:
        self.mha.set_params(params['mha'])
        self.ln1.set_params(params['ln1'])
        self.ln2.set_params(params['ln2'])
        self.ffn.set_params(params['ffn'])

class JazzTransformer:
    """
    Decoder-only Transformer

    Args:
        input_size: Dimension of concatenated embedding input (= TOTAL_EMBED_SIZE)
        hidden_size: Internal model dimension d_model         (= TRANSFORMER_HIDDEN_SIZE)
        num_heads: Number of attention heads                 (= NUM_HEADS)
        num_blocks: Number of transformer blocks             (= NUM_BLOCKS)
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = config.NUM_HEADS, num_blocks: int = config.NUM_BLOCKS):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        
        self.embeddings: Dict[str, Embedding] = {
            emb_key: Embedding(emb_dim, hidden_size) for emb_key, emb_dim in config.EMBEDDING_DIMS.items()
        }
        self.blocks: List[TransformerBlock] = [TransformerBlock(hidden_size, num_heads) for _ in range(num_blocks)]
        self.lm_head = Linear(hidden_size, input_size)
        
    def forward(self, x: np.ndarray):
        """
        Args:
            x: Input, shape (batch_size, seq_len, input_size)
        Returns:
            y: Output, shape (batch_size, seq_len, hidden_size)
        """
        emb_x = np.concatenate([emb(x[..., i]) for i, emb in enumerate(self.embeddings.values())], axis=-1)
        for block in self.blocks:
            emb_x = block.forward(emb_x)
        return self.lm_head.forward(emb_x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the transformer.
        
        Args:
            grad_output: Gradient of the output, shape (batch_size, seq_len, hidden_size)
        Returns:
            grad_input: Gradient of the input, shape (batch_size, seq_len, input_size)
        """
        grad_input = self.lm_head.backward(grad_output)
        for block in reversed(self.blocks):
            grad_input = block.backward(grad_input)
        return grad_input
    
    def get_params(self) -> Dict[str, dict]:
        params: Dict[str, dict] = {}
        for i, block in enumerate(self.blocks):
            params[f'block_{i}'] = block.get_params()
        params['lm_head'] = self.lm_head.get_params()
        return params

    def get_grads(self) -> Dict[str, dict]:
        grads: Dict[str, dict] = {}
        for i, block in enumerate(self.blocks):
            grads[f'block_{i}'] = block.get_grads()
        grads['lm_head'] = self.lm_head.get_grads()
        return grads

    def set_params(self, params: dict) -> None:
        for i, block in enumerate(self.blocks):
            key = f'block_{i}'
            if key in params:
                block.set_params(params[key])
        if 'lm_head' in params:
            self.lm_head.set_params(params['lm_head'])

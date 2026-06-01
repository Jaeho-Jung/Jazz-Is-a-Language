"""
Jazz Solo Generator - Decoder-Only Transformer (GPT-style)

Architecture: All 7 features → embeddings → causal self-attention blocks → output heads
Same feature interface as RNN/LSTM models for consistency.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.Transformer_pytorch import config


# =============================================================================
# ACTIVATION
# =============================================================================

class NewGELU(nn.Module):
    """GELU activation (Google BERT / OpenAI GPT version)."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# =============================================================================
# ATTENTION
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=512):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, _ = x.size()
        
        # Project and reshape: (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, n_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        
        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_dropout(self.out_proj(out))
        
        return out


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """Causal transformer block: LayerNorm → Self-Attention → LayerNorm → FFN."""
    
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            NewGELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================

class JazzTransformer(nn.Module):
    """
    Decoder-Only Transformer (GPT-style) for Jazz Solo Generation.
    
    Input Features (same 7 as RNN/LSTM):
        - pitch: 0-127 (MIDI), 128 (rest)
        - rel_pitch: 0-11 (pitch class relative to chord root), 12 (rest)
        - duration: dynamic vocab from dataset
        - prev_interval: -12 to +12 mapped to 0-24
        - chord_root: 0-11 (pitch class), 12 (NC)
        - chord_quality: 0-5, 6 (NC)
        - metric_pos: 0-47 (position in bar)
    
    Output:
        - pitch_logits: (batch, 129)
        - dur_logits: (batch, num_dur_classes)
    """

    def __init__(self, num_dur_classes):
        super().__init__()
        
        d_model = config.D_MODEL
        n_heads = config.N_HEADS
        n_layers = config.NUM_LAYERS
        dropout = config.DROPOUT
        max_seq_len = config.SEQ_LEN
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # =====================================================================
        # FEATURE EMBEDDINGS (all 7 features)
        # =====================================================================
        self.emb_pitch = nn.Embedding(config.VOCAB_SIZE_PITCH, config.EMBED_SIZE_PITCH)
        self.emb_rel_pitch = nn.Embedding(config.VOCAB_SIZE_REL_PITCH, config.EMBED_SIZE_REL_PITCH)
        self.emb_duration = nn.Embedding(num_dur_classes, config.EMBED_SIZE_DURATION)
        self.emb_prev_interval = nn.Embedding(25, config.EMBED_SIZE_PREV_INTERVAL)
        self.emb_chord_root = nn.Embedding(config.VOCAB_SIZE_CHORD_ROOT, config.EMBED_SIZE_CHORD_ROOT)
        self.emb_chord_quality = nn.Embedding(config.VOCAB_SIZE_CHORD_QUALITY, config.EMBED_SIZE_CHORD_QUALITY)
        self.emb_metric_pos = nn.Embedding(config.VOCAB_SIZE_METRIC_POS, config.EMBED_SIZE_GRID_POS)
        
        # Total embedding dimension
        total_embed_dim = (
            config.EMBED_SIZE_PITCH +
            config.EMBED_SIZE_REL_PITCH +
            config.EMBED_SIZE_DURATION +
            config.EMBED_SIZE_PREV_INTERVAL +
            config.EMBED_SIZE_CHORD_ROOT +
            config.EMBED_SIZE_CHORD_QUALITY +
            config.EMBED_SIZE_GRID_POS
        )
        
        # Project concatenated embeddings to d_model
        self.input_proj = nn.Linear(total_embed_dim, d_model)
        
        # Positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # TRANSFORMER BLOCKS
        # =====================================================================
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, max_seq_len) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        # =====================================================================
        # OUTPUT HEADS
        # =====================================================================
        self.pitch_head = nn.Linear(d_model, config.VOCAB_SIZE_PITCH)
        self.dur_head = nn.Linear(d_model, num_dur_classes)
        
        self.num_dur_classes = num_dur_classes
        
        # Initialize weights
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for name, p in self.named_parameters():
            if 'out_proj.weight' in name or name.endswith('mlp.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"JazzTransformer parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, features, targets=None):
        """
        GPT-style forward pass: predict next token at EVERY position.
        
        Args:
            features: dict with 7 feature tensors, each (batch, seq_len)
            targets: optional dict with 'pitch' and 'duration' tensors,
                     each (batch, seq_len) — shifted targets for all positions
            
        Returns:
            pitch_logits: (batch, seq_len, 129)
            dur_logits: (batch, seq_len, num_dur_classes)
            loss: scalar if targets provided, else None
        """
        device = features['pitch'].device
        B, T = features['pitch'].size()
        
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        # Embed all features
        emb = torch.cat([
            self.emb_pitch(features['pitch']),
            self.emb_rel_pitch(features['rel_pitch']),
            self.emb_duration(features['duration']),
            self.emb_prev_interval(features['prev_interval']),
            self.emb_chord_root(features['chord_root']),
            self.emb_chord_quality(features['chord_quality']),
            self.emb_metric_pos(features['metric_pos']),
        ], dim=-1)  # (B, T, total_embed_dim)
        
        # Project to d_model and add positional embedding
        x = self.input_proj(emb)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.input_dropout(x + self.pos_embed(pos))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Predict at ALL positions (GPT-style)
        pitch_logits = self.pitch_head(x)   # (B, T, 129)
        dur_logits = self.dur_head(x)       # (B, T, num_dur_classes)
        
        # Compute loss across all positions if targets provided
        loss = None
        if targets is not None:
            # Reshape: (B, T, vocab) -> (B*T, vocab), targets: (B, T) -> (B*T,)
            loss_pitch = F.cross_entropy(
                pitch_logits.view(-1, pitch_logits.size(-1)),
                targets['pitch'].view(-1),
                label_smoothing=config.LABEL_SMOOTHING
            )
            loss_dur = F.cross_entropy(
                dur_logits.view(-1, dur_logits.size(-1)),
                targets['duration'].view(-1),
                label_smoothing=config.LABEL_SMOOTHING
            )
            loss = loss_pitch + loss_dur
        
        return pitch_logits, dur_logits, loss

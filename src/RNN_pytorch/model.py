"""
RNN Jazz Solo Generator

Simplified 7-feature architecture optimized for small dataset.
Features: pitch, rel_pitch, duration, prev_interval, chord_root, chord_quality, metric_pos
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from src.RNN_pytorch import config


class JazzRNN(nn.Module):
    """
    Vanilla RNN model for jazz solo generation.
    
    Input Features (7 total):
        - pitch: MIDI pitch (0-127) or rest (128)
        - rel_pitch: Pitch class relative to chord root (0-11) or rest (12)
        - duration: Quantized duration (dynamic vocab)
        - prev_interval: Previous melodic interval (-12 to +12 → 0-24)
        - chord_root: Current chord root (0-11) or NC (12)
        - chord_quality: Chord quality (0-5) or NC (6)
        - metric_pos: Position in bar (0-47)
    
    Output:
        - pitch_logits: (batch, 129)
        - dur_logits: (batch, num_dur_classes)
    """

    def __init__(self, num_dur_classes):
        super(JazzRNN, self).__init__()
        
        # =====================================================================
        # EMBEDDINGS (7 features, 136 total dims)
        # =====================================================================
        self.pitch_embed = nn.Embedding(config.VOCAB_SIZE_PITCH, config.EMBED_SIZE_PITCH)
        self.rel_pitch_embed = nn.Embedding(config.VOCAB_SIZE_REL_PITCH, config.EMBED_SIZE_REL_PITCH)
        self.dur_embed = nn.Embedding(num_dur_classes, config.EMBED_SIZE_DURATION)
        self.prev_int_embed = nn.Embedding(config.VOCAB_SIZE_PREV_INTERVAL, config.EMBED_SIZE_PREV_INTERVAL)
        self.chord_root_embed = nn.Embedding(config.VOCAB_SIZE_CHORD_ROOT, config.EMBED_SIZE_CHORD_ROOT)
        self.chord_qual_embed = nn.Embedding(config.VOCAB_SIZE_CHORD_QUALITY, config.EMBED_SIZE_CHORD_QUALITY)
        self.metric_pos_embed = nn.Embedding(config.VOCAB_SIZE_METRIC_POS, config.EMBED_SIZE_METRIC_POS)
        
        # Calculate input size
        input_size = (
            config.EMBED_SIZE_PITCH +
            config.EMBED_SIZE_REL_PITCH +
            config.EMBED_SIZE_DURATION +
            config.EMBED_SIZE_PREV_INTERVAL +
            config.EMBED_SIZE_CHORD_ROOT +
            config.EMBED_SIZE_CHORD_QUALITY +
            config.EMBED_SIZE_METRIC_POS
        )  # = 136
        
        # =====================================================================
        # RNN BACKBONE
        # =====================================================================
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        # =====================================================================
        # OUTPUT HEADS
        # =====================================================================
        self.pitch_head = nn.Linear(config.HIDDEN_SIZE, config.VOCAB_SIZE_PITCH)
        self.dur_head = nn.Linear(config.HIDDEN_SIZE, num_dur_classes)
        
        self.num_dur_classes = num_dur_classes
        
        # Initialize weights
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"JazzRNN parameters: {n_params/1e6:.2f}M")

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'embed' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, features, targets=None):
        """
        Args:
            features: dict with keys 'pitch', 'rel_pitch', 'duration', 
                     'prev_interval', 'chord_root', 'chord_quality', 'metric_pos'
            targets: optional dict with 'pitch' and 'duration'
        Returns:
            pitch_logits, dur_logits, loss (if targets provided)
        """
        # Embed all features
        emb_pitch = self.pitch_embed(features['pitch'])
        emb_rel = self.rel_pitch_embed(features['rel_pitch'])
        emb_dur = self.dur_embed(features['duration'])
        emb_pint = self.prev_int_embed(features['prev_interval'])
        emb_cr = self.chord_root_embed(features['chord_root'])
        emb_cq = self.chord_qual_embed(features['chord_quality'])
        emb_pos = self.metric_pos_embed(features['metric_pos'])
        
        # Concatenate: (batch, seq_len, 136)
        rnn_input = torch.cat([
            emb_pitch, emb_rel, emb_dur, emb_pint, emb_cr, emb_cq, emb_pos
        ], dim=-1)
        
        # RNN forward (note: RNN returns (output, h_n), not (output, (h_n, c_n)))
        output, h_n = self.rnn(rnn_input)
        
        # Take last timestep
        last_hidden = output[:, -1, :]
        
        # Output heads
        pitch_logits = self.pitch_head(last_hidden)
        dur_logits = self.dur_head(last_hidden)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss_pitch = F.cross_entropy(pitch_logits, targets['pitch'])
            loss_dur = F.cross_entropy(dur_logits, targets['duration'])
            loss = loss_pitch + loss_dur
        
        return pitch_logits, dur_logits, loss

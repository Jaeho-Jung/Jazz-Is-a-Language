"""
Generator for JazzTransformer (Decoder-Only) model.

Supports top-k sampling for better generation quality.

Features: pitch, rel_pitch, duration, prev_interval, 
chord_root, chord_quality, metric_pos
"""

import torch
import numpy as np
from src.Transformer_pytorch import config


class JazzGenerator:
    def __init__(self, model, dataset, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.model.eval()
        
    def generate(self, chord_progression, temperature=1.0, top_k=10):
        """
        Autoregressive generation given a chord progression.
        
        Args:
            chord_progression: List of (root, quality, duration_grid).
            temperature: Sampling temperature (lower = more focused).
            top_k: Only sample from top-k most likely tokens. None = no filtering.
        Returns:
            List of (pitch, duration, start_pos_abs)
        """
        # Build timeline
        timeline = []
        current_abs = 0
        for root, qual, dur in chord_progression:
            timeline.append({
                'start': current_abs,
                'end': current_abs + dur,
                'root': root,
                'qual': qual
            })
            current_abs += dur
        total_duration = current_abs
        
        # Initialize state
        current_pos_abs = 0
        generated_events = []
        
        # History for 7 features
        history = {
            'pitch': [128],           # Start with rest
            'rel_pitch': [12],        # Rest -> 12
            'duration': [0],          # Default duration idx
            'prev_interval': [12],    # 0 interval -> idx 12
            'chord_root': [],
            'chord_quality': [],
            'metric_pos': [0],
        }
        
        # Fill chord info for start position
        c_root, c_qual = self._get_chord_at_pos(0, timeline)
        history['chord_root'].append(c_root)
        history['chord_quality'].append(c_qual)
        
        while current_pos_abs < total_duration:
            # Prepare input (last SEQ_LEN items)
            seq_len = min(len(history['pitch']), config.SEQ_LEN)
            
            features = {}
            for k in history:
                data = history[k][-seq_len:]
                features[k] = torch.LongTensor([data]).to(self.device)
            
            # Forward (no targets -> no loss)
            with torch.no_grad():
                pitch_logits, dur_logits, _ = self.model(features, None)
                
            # Take LAST position's logits (GPT-style: position T-1 predicts token T)
            pitch_logits = pitch_logits[:, -1, :]  # (1, vocab)
            dur_logits = dur_logits[:, -1, :]      # (1, vocab)
                
            # Sample with temperature + top-k
            next_pitch = self._sample(pitch_logits, temperature, top_k)
            next_dur_idx = self._sample(dur_logits, temperature, top_k)
            
            # Decode duration
            next_dur_val = self.dataset.idx_to_dur.get(next_dur_idx, 12)
            
            # Append to result
            generated_events.append((next_pitch, next_dur_val, current_pos_abs))
            
            # Update position
            current_pos_abs += next_dur_val
            
            # Update history for next step
            history['pitch'].append(next_pitch)
            history['duration'].append(next_dur_idx)
            
            # Metric position
            next_metric_pos = current_pos_abs % config.VOCAB_SIZE_METRIC_POS
            history['metric_pos'].append(next_metric_pos)
            
            # Chord info at new position
            c_root, c_qual = self._get_chord_at_pos(current_pos_abs, timeline)
            history['chord_root'].append(c_root)
            history['chord_quality'].append(c_qual)
            
            # Relative pitch
            if next_pitch < 128 and c_root < 12:
                rel = (next_pitch - c_root) % 12
            else:
                rel = 12
            history['rel_pitch'].append(rel)
            
            # Prev interval
            prev_p = history['pitch'][-2]
            if next_pitch < 128 and prev_p < 128:
                interval = next_pitch - prev_p
            else:
                interval = 0
            interval_idx = int(np.clip(interval + 12, 0, 24))
            history['prev_interval'].append(interval_idx)
            
        return generated_events

    def _sample(self, logits, temperature=1.0, top_k=None):
        """Sample from logits with temperature and optional top-k filtering."""
        logits = logits / temperature
        
        if top_k is not None:
            # Zero out everything outside top-k
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def _get_chord_at_pos(self, abs_pos, timeline):
        """Get chord root and quality at given position."""
        for segment in timeline:
            if segment['start'] <= abs_pos < segment['end']:
                return segment['root'], segment['qual']
        return 12, 6  # NC (no chord)

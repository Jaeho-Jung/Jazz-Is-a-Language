"""
Chord Parsing and Harmony Processing

Handles chord symbol parsing, quality mapping, and harmonic calculations.
"""

import pandas as pd
from typing import Optional, Tuple
from .config import (
    CHORD_QUALITY_MAP,
    NOTE_NAMES,
    NOTE_NAMES_FLAT,
    QUALITY_SUS,
    QUALITY_DOM
)


class ChordParser:
    """Parser for jazz chord symbols."""
    
    @staticmethod
    def parse_chord_root(chord_str: str) -> Optional[int]:
        """
        Extract chord root from chord string and convert to pitch class (0-11).
        
        Args:
            chord_str: Chord symbol (e.g., 'Cmaj7', 'G7', 'Dm7b5')
        
        Returns:
            Pitch class (0-11) or None if parsing fails
        """
        if not chord_str or pd.isna(chord_str):
            return None
        
        chord_str = str(chord_str).strip()
        
        # Try to match note name at the beginning
        # Handle both sharp (#) and flat (b) notation
        if len(chord_str) >= 2 and chord_str[1] in ['#', 'b']:
            root_str = chord_str[:2]
        elif len(chord_str) >= 1:
            root_str = chord_str[0]
        else:
            return None
        
        # Convert to pitch class
        try:
            if root_str in NOTE_NAMES:
                return NOTE_NAMES.index(root_str)
            elif root_str in NOTE_NAMES_FLAT:
                return NOTE_NAMES_FLAT.index(root_str)
        except:
            pass
        
        return None
    
    @staticmethod
    def parse_chord_quality(chord_str: str) -> int:
        """
        Extract chord quality from chord string and map to quality index.
        
        Quality mapping:
        0: Maj
        1: min
        2: dom
        3: half-dim
        4: dim
        5: sus/other
        
        Args:
            chord_str: Chord symbol (e.g., 'Cmaj7', 'G7', 'Dm7b5')
        
        Returns:
            Quality index (0-5)
        """
        if not chord_str or pd.isna(chord_str):
            return QUALITY_SUS  # Default to 'sus/other'
        
        chord_str = str(chord_str).strip().lower()
        
        # Remove root note
        if len(chord_str) >= 2 and chord_str[1] in ['#', 'b']:
            quality_str = chord_str[2:]
        elif len(chord_str) >= 1:
            quality_str = chord_str[1:]
        else:
            return QUALITY_SUS
        
        # Check each quality category
        for quality_idx, (quality_name, patterns) in enumerate(CHORD_QUALITY_MAP.items()):
            for pattern in patterns:
                if pattern.lower() in quality_str:
                    return quality_idx
        
        return QUALITY_SUS  # Default to 'sus/other'
    
    @staticmethod
    def calculate_secondary_dominant(target_root: int) -> Tuple[int, int]:
        """
        Calculate the secondary dominant (V7) of a target chord.
        Used for filling pickup measure chords.
        
        Args:
            target_root: Target chord root (0-11)
        
        Returns:
            Tuple of (dominant_root, quality_index)
            quality_index is 2 (dom)
        """
        # V7 is a perfect fifth above the target (7 semitones)
        dominant_root = (target_root + 7) % 12
        return dominant_root, QUALITY_DOM
    
    @staticmethod
    def key_to_pitch_class(key_str: str) -> int:
        """
        Convert key string to pitch class (0-11).
        
        Args:
            key_str: Key string (e.g., 'C', 'Bb', 'F#')
        
        Returns:
            Pitch class (0-11), defaults to 0 (C) if parsing fails
        """
        if not key_str or pd.isna(key_str):
            return 0
        
        key_str = str(key_str).strip()
        
        # Extract just the note name (ignore major/minor designation)
        if len(key_str) >= 2 and key_str[1] in ['#', 'b']:
            note_str = key_str[:2]
        elif len(key_str) >= 1:
            note_str = key_str[0]
        else:
            return 0
        
        if note_str in NOTE_NAMES:
            return NOTE_NAMES.index(note_str)
        elif note_str in NOTE_NAMES_FLAT:
            return NOTE_NAMES_FLAT.index(note_str)
        
        return 0

"""
Chord Parsing and Harmony Processing
"""

import pandas as pd
from typing import Optional, Tuple, Dict
from .config import NOTE_NAMES, NOTE_NAMES_SHARP, CHORD_QUALITY_MAP

class ChordParser:
    """Parser for jazz chord symbols."""

    @staticmethod
    def _is_valid_chord(chord_symbol: str) -> bool:
        """
        Check if the given string is a valid chord symbol.

        Args:
            chord_symbol (str): String to check

        Returns:
            bool: True if the string is a valid chord symbol, False otherwise
        """
        return chord_symbol != 'NC' and chord_symbol != ''

    @staticmethod
    def parse_chord(chord_symbol: str) -> Optional[Dict[str, int]]:
        """
        Extract chord root and quality from chord symbol.
        Then convert to pitch class (0-11) and quality index (0-3).

        Quality mapping:
        0: Maj
        1: min
        2: dom
        3: half-dim

        Args:
            chord_symbol (str): Chord symbol (e.g., 'Ebj7', 'F-7', 'Gm7b5', 'F', 'Bb7913')

        Returns:
            Optional[int, int]: Pitch class of the chord root and quality index (0-3)
        """
        if not ChordParser._is_valid_chord(chord_symbol):
            return None
        
        chord_symbol = chord_symbol.strip()
        
        if len(chord_symbol) >= 2 and chord_symbol[1] in ['#', 'b']:
            root = chord_symbol[:2]
            quality = chord_symbol[2:]
        else:
            root = chord_symbol[:1]
            quality = chord_symbol[1:]
        
        # Convert root to pitch class (initialize to 0 as default)
        root_idx = 0
        try:
            if root in NOTE_NAMES:
                root_idx = NOTE_NAMES.index(root)
            elif root in NOTE_NAMES_SHARP:
                root_idx = NOTE_NAMES_SHARP.index(root)
        except:
            pass

        # Convert quality to quality index (major as default)
        quality_idx = 0
        try:
            for idx, key in enumerate(CHORD_QUALITY_MAP):
                if quality in CHORD_QUALITY_MAP[key]:
                    quality_idx = idx
                    break
        except:
            pass

        return {
            'root': int(root_idx),
            'quality': int(quality_idx)
        }

    @staticmethod
    def calculate_secondary_dominant(target_root: int) -> Tuple[int, int]:
        """
        Calculate secondary dominant of a target chord.

        Args:
            target_root: Target chord root (0-11)

        Returns:
            Tuple[int, int]: (dominant_root, dominant_quality_index)
        """
        dominant_root = (target_root + 7) % 12
        quality_dom = list(CHORD_QUALITY_MAP).index('dom')
        return dominant_root, quality_dom

    
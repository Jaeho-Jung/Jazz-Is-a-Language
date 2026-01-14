"""
Data Augmentation Strategies

Handles transposition and other data augmentation techniques.
"""

import pandas as pd
from typing import List
from .config import MIDI_MIN, MIDI_MAX, AUGMENTATION_SHIFTS
from .utils import clip_to_range


class DataAugmentor:
    """Handles data augmentation for solo data."""
    
    @staticmethod
    def transpose_solo(processed_df: pd.DataFrame, shift: int) -> pd.DataFrame:
        """
        Transpose a solo by a given number of semitones.
        
        Args:
            processed_df: Processed solo DataFrame
            shift: Number of semitones to transpose (-5 to +6)
        
        Returns:
            Augmented DataFrame
        """
        aug_df = processed_df.copy()
        
        # Transpose pitch
        aug_df['pitch_normalized'] = aug_df['pitch_normalized'] + shift
        
        # Validate MIDI range and clip if necessary
        aug_df['pitch_normalized'] = aug_df['pitch_normalized'].clip(MIDI_MIN, MIDI_MAX)
        
        # Transpose chord roots
        aug_df['chord_root'] = aug_df['chord_root'].apply(
            lambda x: (x + shift) % 12 if pd.notna(x) else x
        )
        aug_df['next_chord_root'] = aug_df['next_chord_root'].apply(
            lambda x: (x + shift) % 12 if pd.notna(x) else x
        )
        
        # Transpose key center
        aug_df['key_center'] = (aug_df['key_center'] + shift) % 12
        
        # Recalculate chord-relative pitch
        aug_df['chord_rel_pitch'] = aug_df.apply(
            lambda row: (row['pitch_normalized'] - row['chord_root']) % 12 
            if pd.notna(row['chord_root']) else 0,
            axis=1
        )
        
        # Note: chord_root_rel and next_chord_root_rel remain the same
        # because both key and chord roots are transposed by the same amount
        
        # Add shift information
        aug_df['key_shift'] = shift
        
        return aug_df
    
    @staticmethod
    def validate_transposition(df: pd.DataFrame) -> bool:
        """
        Validate that transposed pitches are within MIDI range.
        
        Args:
            df: Transposed DataFrame
        
        Returns:
            True if valid, False otherwise
        """
        if 'pitch_normalized' not in df.columns:
            return False
        
        min_pitch = df['pitch_normalized'].min()
        max_pitch = df['pitch_normalized'].max()
        
        return MIDI_MIN <= min_pitch <= MIDI_MAX and MIDI_MIN <= max_pitch <= MIDI_MAX
    
    def augment_dataset(
        self, 
        processed_solos: List[pd.DataFrame],
        shifts: List[int] = None
    ) -> List[pd.DataFrame]:
        """
        Apply augmentation pipeline to all solos.
        
        Args:
            processed_solos: List of processed solo DataFrames
            shifts: List of semitone shifts to apply (default: AUGMENTATION_SHIFTS)
        
        Returns:
            List of augmented solo DataFrames
        """
        if shifts is None:
            shifts = AUGMENTATION_SHIFTS
        
        augmented_solos = []
        
        for solo_df in processed_solos:
            for shift in shifts:
                if shift == 0:
                    # Original already has key_shift = 0
                    if 'key_shift' not in solo_df.columns:
                        solo_df['key_shift'] = 0
                    augmented_solos.append(solo_df)
                else:
                    aug_df = self.transpose_solo(solo_df, shift)
                    augmented_solos.append(aug_df)
        
        return augmented_solos

"""
Feature Engineering and Dataset Creation

Extracts features from processed solo data and creates training datasets.
"""

import pandas as pd
from typing import List, Dict
from .chord_parser import ChordParser
from .rhythm_processor import RhythmProcessor


class FeatureExtractor:
    """Extracts features from solo data for model training."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.chord_parser = ChordParser()
        self.rhythm_processor = RhythmProcessor()
    
    def process_solo(self, solo_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process a single solo: align chords, calculate features, etc.
        
        Args:
            solo_data: Dictionary with melody, beats, and solo_info DataFrames
        
        Returns:
            Processed DataFrame with all features
        """
        melody_df = solo_data['melody'].copy()
        solo_info = solo_data['solo_info'].iloc[0]
        
        # Get key center
        key_center = self.chord_parser.key_to_pitch_class(solo_info['key'])
        
        # Normalize pitch to nearest integer
        melody_df['pitch_normalized'] = melody_df['pitch'].round().astype(int)
        
        # Calculate position grid
        melody_df['pos_grid'] = melody_df.apply(
            lambda row: self.rhythm_processor.calculate_position_grid(
                row['beat'], row['tatum'], row['division']
            ),
            axis=1
        )
        
        # Calculate duration grid
        melody_df['dur_grid'] = melody_df.apply(
            lambda row: self.rhythm_processor.calculate_duration_grid(
                row['duration'], row['beatdur']
            ),
            axis=1
        )
        
        # Calculate interval from previous note
        melody_df['prev_interval'] = melody_df['pitch_normalized'].diff().fillna(0).astype(int)
        
        # Calculate chord-relative pitch
        melody_df['chord_rel_pitch'] = melody_df.apply(
            lambda row: (row['pitch_normalized'] - row['chord_root']) % 12 
            if pd.notna(row['chord_root']) else 0,
            axis=1
        )
        
        # Add key center
        melody_df['key_center'] = key_center
        
        # Calculate chord root relative to key
        melody_df['chord_root_rel'] = melody_df.apply(
            lambda row: (row['chord_root'] - key_center) % 12 
            if pd.notna(row['chord_root']) else 0,
            axis=1
        )
        
        # Calculate next chord root relative to key
        melody_df['next_chord_root_rel'] = melody_df.apply(
            lambda row: (row['next_chord_root'] - key_center) % 12 
            if pd.notna(row['next_chord_root']) else 0,
            axis=1
        )
        
        # Add metadata
        melody_df['performer'] = solo_info['performer']
        melody_df['title'] = solo_info['title']
        melody_df['style'] = solo_info['style']
        melody_df['avgtempo'] = solo_info['avgtempo']
        
        return melody_df
    
    def create_training_dataset(self, processed_solos: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Create final training dataset with input features (X) and target labels (Y).
        
        Args:
            processed_solos: List of processed and augmented solo DataFrames
        
        Returns:
            Final training DataFrame
        """
        all_data = []
        
        for solo_df in processed_solos:
            melid = solo_df['melid'].iloc[0]
            key_shift = solo_df['key_shift'].iloc[0] if 'key_shift' in solo_df.columns else 0
            
            for idx in range(len(solo_df)):
                row = solo_df.iloc[idx]
                
                # Prepare feature dictionary
                features = {
                    # Identifiers (for debugging)
                    'melid': melid,
                    'key_shift': key_shift,
                    'step': idx,
                    
                    # Input features (X)
                    'chord_root_rel': row['chord_root_rel'],
                    'chord_quality': row['chord_quality'],
                    'next_chord_root_rel': row['next_chord_root_rel'],
                    'next_chord_quality': row['next_chord_quality'],
                    'pos_grid': row['pos_grid'],
                    'prev_interval': row['prev_interval'],
                    'chord_rel_pitch': row['chord_rel_pitch'],
                    
                    # Target labels (Y)
                    'target_pitch': row['pitch_normalized'],
                    'target_dur': row['dur_grid'],
                    
                    # Additional context (optional)
                    'bar': row['bar'],
                    'beat': row['beat'],
                    'key_center': row['key_center'],
                    
                    # Metadata
                    'performer': row['performer'],
                    'title': row['title'],
                    'style': row['style'],
                    'avgtempo': row['avgtempo']
                }
                
                all_data.append(features)
        
        final_df = pd.DataFrame(all_data)
        
        return final_df

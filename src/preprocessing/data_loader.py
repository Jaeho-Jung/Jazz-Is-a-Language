"""
Database Loading and Data Extraction

Handles SQLite database operations, data loading, and chord alignment.
"""

import sqlite3
import pandas as pd
from typing import List, Dict
from .config import TARGET_MELODY_TYPE, TARGET_STYLES, TARGET_SIGNATURE, NOTE_NAMES
from .chord_parser import ChordParser


class WJDDatabase:
    """Manager for Weimar Jazz Database operations."""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.chord_parser = ChordParser()
    
    def get_target_melids(self) -> List[int]:
        """
        Extract melids that match the filtering criteria:
        - melody_type.type = 'SOLO'
        - solo_info.style in ['BEBOP', 'HARDBOP']
        - beats.signature = '4/4'
        
        Returns:
            List of melids
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT DISTINCT m.melid
        FROM melody_type m
        JOIN solo_info s ON m.melid = s.melid
        JOIN beats b ON m.melid = b.melid
        WHERE m.type = ?
          AND s.style IN (?, ?)
          AND b.signature = ?
        ORDER BY m.melid
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(TARGET_MELODY_TYPE, TARGET_STYLES[0], TARGET_STYLES[1], TARGET_SIGNATURE)
        )
        
        conn.close()
        
        melids = df['melid'].tolist()
        print(f"Found {len(melids)} solos matching criteria: {TARGET_STYLES}, {TARGET_SIGNATURE}")
        
        return melids
    
    def load_solo_data(self, melid: int) -> Dict[str, pd.DataFrame]:
        """
        Load all relevant data for a single solo.
        
        Args:
            melid: Melody ID
        
        Returns:
            Dictionary containing DataFrames for melody, beats, and solo_info
        """
        conn = sqlite3.connect(self.db_path)
        
        # Load melody events
        melody_query = """
        SELECT eventid, melid, onset, pitch, duration, 
               bar, beat, tatum, division, beatdur
        FROM melody
        WHERE melid = ?
        ORDER BY eventid
        """
        melody_df = pd.read_sql_query(melody_query, conn, params=(melid,))
        
        # Load beat/chord information
        beats_query = """
        SELECT beatid, melid, bar, beat, chord, form, chorus_id
        FROM beats
        WHERE melid = ?
        ORDER BY bar, beat
        """
        beats_df = pd.read_sql_query(beats_query, conn, params=(melid,))
        
        # Load solo metadata
        solo_query = """
        SELECT melid, key, avgtempo, signature, style, performer, title
        FROM solo_info
        WHERE melid = ?
        """
        solo_df = pd.read_sql_query(solo_query, conn, params=(melid,))
        
        conn.close()
        
        return {
            'melody': melody_df,
            'beats': beats_df,
            'solo_info': solo_df
        }
    
    def align_chords_to_melody(self, melody_df: pd.DataFrame, beats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align chord information from beats table to each melody event.
        Also adds 'next_chord' lookahead information.
        
        Args:
            melody_df: Melody events DataFrame
            beats_df: Beats DataFrame with chord information
        
        Returns:
            Melody DataFrame with chord columns added
        """
        # Parse chords in beats_df
        beats_df['chord_root'] = beats_df['chord'].apply(self.chord_parser.parse_chord_root)
        beats_df['chord_quality'] = beats_df['chord'].apply(self.chord_parser.parse_chord_quality)
        
        # Create a mapping from (bar, beat) to chord info
        chord_map = {}
        for _, row in beats_df.iterrows():
            key = (row['bar'], row['beat'])
            chord_map[key] = {
                'chord': row['chord'],
                'chord_root': row['chord_root'],
                'chord_quality': row['chord_quality']
            }
        
        # Align chords to melody events
        melody_df['chord'] = None
        melody_df['chord_root'] = None
        melody_df['chord_quality'] = None
        
        for idx, row in melody_df.iterrows():
            bar = row['bar']
            beat = row['beat']
            
            # Find the chord at this bar/beat
            # Look backwards if exact match not found
            chord_info = None
            for b in range(int(beat), 0, -1):
                if (bar, b) in chord_map:
                    chord_info = chord_map[(bar, b)]
                    break
            
            if chord_info:
                melody_df.at[idx, 'chord'] = chord_info['chord']
                melody_df.at[idx, 'chord_root'] = chord_info['chord_root']
                melody_df.at[idx, 'chord_quality'] = chord_info['chord_quality']
        
        # Add next chord lookahead
        melody_df['next_chord_root'] = None
        melody_df['next_chord_quality'] = None
        
        for idx in range(len(melody_df) - 1):
            current_chord_root = melody_df.at[idx, 'chord_root']
            current_chord_quality = melody_df.at[idx, 'chord_quality']
            
            # Look ahead to find the next different chord
            for next_idx in range(idx + 1, len(melody_df)):
                next_chord_root = melody_df.at[next_idx, 'chord_root']
                next_chord_quality = melody_df.at[next_idx, 'chord_quality']
                
                if (next_chord_root != current_chord_root or 
                    next_chord_quality != current_chord_quality):
                    melody_df.at[idx, 'next_chord_root'] = next_chord_root
                    melody_df.at[idx, 'next_chord_quality'] = next_chord_quality
                    break
        
        # For the last event, use the same chord as current
        if len(melody_df) > 0:
            last_idx = len(melody_df) - 1
            melody_df.at[last_idx, 'next_chord_root'] = melody_df.at[last_idx, 'chord_root']
            melody_df.at[last_idx, 'next_chord_quality'] = melody_df.at[last_idx, 'chord_quality']
        
        return melody_df
    
    def handle_pickup_measures(self, melody_df: pd.DataFrame, beats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle pickup measures (anacrusis) by filling missing chords with
        secondary dominant of the first valid chord.
        
        Args:
            melody_df: Melody DataFrame with chord alignment
            beats_df: Beats DataFrame
        
        Returns:
            Melody DataFrame with pickup chords filled
        """
        # Find first valid chord
        first_valid_idx = melody_df['chord_root'].first_valid_index()
        
        if first_valid_idx is None:
            return melody_df
        
        # Check if there are events before the first valid chord
        if first_valid_idx > 0:
            first_chord_root = melody_df.at[first_valid_idx, 'chord_root']
            
            # Calculate secondary dominant
            dominant_root, dominant_quality = self.chord_parser.calculate_secondary_dominant(
                int(first_chord_root)
            )
            
            # Fill pickup measures
            for idx in range(first_valid_idx):
                if pd.isna(melody_df.at[idx, 'chord_root']):
                    melody_df.at[idx, 'chord_root'] = dominant_root
                    melody_df.at[idx, 'chord_quality'] = dominant_quality
                    melody_df.at[idx, 'chord'] = f"{NOTE_NAMES[dominant_root]}7"
        
        return melody_df

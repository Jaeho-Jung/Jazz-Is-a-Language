from pathlib import Path
from typing import Dict

import sqlite3
import pandas as pd

from .config import DB_PATH, OUTPUT_PKL, OUTPUT_CSV, GRID_PER_BEAT, GRID_PER_BAR, NOTE_NAMES
from .data_loader import WJD
from .chord_parser import ChordParser
from .utils import safe_int, safe_divide, clip_to_range


class Preprocessor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.chord_parser = ChordParser()
    
    def process(self) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            db_path (str): Path to the database file
        """

        db = WJD(self.db_path)
        
        # Step 1: Extract target melids
        melids = db.get_target_melids()
        
        # Step 2: Process each solo
        all_solos = []
        for melid in melids:
            solo_df = self._process_single_solo(db, melid)
            all_solos.append(solo_df)
            
        dataset = pd.concat(all_solos, ignore_index=True)

        return dataset

    def _process_single_solo(self, db: WJD, melid: int) -> pd.DataFrame:
        """
        Process a single solo through the pipeline.

        Args:
            db (WJD): Database manager
            melid (int): ID of the solo

        Returns:
            pd.DataFrame: Processed solo data
        """
        # Load and flatten
        solo_df = self._load_and_flatten_solo(db, melid)
        
        # Chord processing
        solo_df = self._align_chords(solo_df)
        solo_df = self._fill_pickup_measures(solo_df)
        
        # Rhythmic features
        solo_df = self._add_rhythmic_features(solo_df)
        
        # Melodic features
        solo_df = self._add_melodic_features(solo_df)
        
        # Harmonic features
        solo_df = self._add_harmonic_features(solo_df)
        
        # Select final features
        solo_df = self._select_final_features(solo_df)
        
        # Convert to proper data types
        solo_df = self._convert_dtypes(solo_df)
        
        return solo_df

    def _load_and_flatten_solo(self, db: WJD, melid: int) -> pd.DataFrame:
        """
        Load solo data and flatten into single DataFrame.
        
        Strategy: Use beats as base (has all beat positions), then merge melody data.
        This ensures we have all beat positions, including those without melody events.
        
        Returns a DataFrame with columns:
        - From beats: beatid, bar, beat, chord, signature, chorus_id
        - From melody: eventid, pitch, onset, duration, tatum, division (merged on bar, beat)
        - From solo_info: key, avgtempo, performer, style, title (broadcast to all rows)
        """
        solo_data = db.load_solo_data(melid)

        # Use beats as base (contains ALL beat positions)
        df = solo_data['beats'].copy()

        # Merge melody info on bar+beat
        # Use 'left' join to keep all beats, even those without melody
        melody_subset = solo_data['melody'][['bar', 'beat', 'eventid', 'pitch', 'onset', 'duration', 'tatum', 'division', 'beatdur']].copy()
        df = df.merge(melody_subset, on=['bar', 'beat'], how='left')
        
        # Add solo-level metadata (broadcast to all rows)
        for col in ['key', 'avgtempo', 'performer', 'style', 'title']:
            df[col] = solo_data['solo_info'][col].iloc[0]
        
        # Add melid for reference
        df['melid'] = melid
        
        return df

    def _align_chords(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and align chords. Adds columns:
        - chord_root: int (0-11)
        - chord_quality: int (0-3)
        - next_chord, next_chord_root, next_chord_quality

        Args:
            solo_df (pd.DataFrame): DataFrame containing melody, beats, and solo_info DataFrames

        Returns:
            Solo Data with chord columns added
        """
        solo_df = solo_df.copy()
        
        # Parse chords - parse_chord returns dict or None
        # Use pd.Int64Dtype() to handle None values without converting to float
        parsed_chords = solo_df['chord'].apply(self.chord_parser.parse_chord)
        solo_df['chord_root'] = parsed_chords.apply(lambda x: x['root'] if x else pd.NA).astype(pd.Int64Dtype())
        solo_df['chord_quality'] = parsed_chords.apply(lambda x: x['quality'] if x else pd.NA).astype(pd.Int64Dtype())

        # Align chords to melody events
        idx_last_chord = 0
        for idx, row in solo_df.iterrows():
            if row['chord'] != '':
                idx_last_chord = idx
            solo_df.at[idx, 'chord'] = solo_df.at[idx_last_chord, 'chord']
            solo_df.at[idx, 'chord_root'] = solo_df.at[idx_last_chord, 'chord_root']
            solo_df.at[idx, 'chord_quality'] = solo_df.at[idx_last_chord, 'chord_quality']

        # Add next chord lookahead
        cur_chord = solo_df.iloc[-1]['chord']
        idx_next_chord = solo_df.index[-1]
        for idx, row in solo_df.iloc[::-1].iterrows():
            if row['chord'] != cur_chord:
                idx_next_chord = idx+1
                cur_chord = row['chord']
            solo_df.at[idx, 'next_chord'] = solo_df.at[idx_next_chord, 'chord']
            solo_df.at[idx, 'next_chord_root'] = solo_df.at[idx_next_chord, 'chord_root']
            solo_df.at[idx, 'next_chord_quality'] = solo_df.at[idx_next_chord, 'chord_quality']
        
        return solo_df

    def _fill_pickup_measures(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle pickup measures by filling missing chord with
        secondary dominant of the first valid chord.
        
        Args:
            solo_df (pd.DataFrame): DataFrame containing melody, beats, and solo_info DataFrames
        
        Returns:
            Solo Data with pickup chord filled
        """
        solo_df = solo_df.copy()

        # Find first valid chord
        first_valid_idx = solo_df.index[(solo_df['chord'] != 'NC') & (solo_df['chord'] != '')].min()

        if first_valid_idx is None:
            return solo_df

        # Check if there are events before the first valid chord
        if first_valid_idx > 0:
            first_chord_root = solo_df.at[first_valid_idx, 'chord_root']

            # Calculate secondary dominant
            dominant_root, dominant_quality = self.chord_parser.calculate_secondary_dominant(first_chord_root)

            # Fill pickup measures
            for idx in range(first_valid_idx):
                solo_df.at[idx, 'chord_root'] = dominant_root
                solo_df.at[idx, 'chord_quality'] = dominant_quality
                solo_df.at[idx, 'chord'] = f"{NOTE_NAMES[dominant_root]}7"
        
        return solo_df

    def _add_rhythmic_features(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all rhythmic-related features to the solo data.
        - position grid
        - duration grid
        
        Args:
            solo_df (pd.DataFrame): DataFrame containing melody, beats, and solo_info DataFrames
        
        Returns:
            Solo Data with rhythmic features added
        """
        solo_df = solo_df.copy()

        # Calculate position grid
        solo_df['pos_grid'] = solo_df.apply(
            lambda row: self._calculate_position_grid(
                row['beat'], row['tatum'], row['division']
            ),
            axis=1
        )
        
        # Calculate duration grid
        solo_df['dur_grid'] = solo_df.apply(
            lambda row: self._calculate_duration_grid(
                row['duration'], row['beatdur']
            ),
            axis=1
        )

        return solo_df
    
    def _add_melodic_features(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add melodic-related features to the solo data.
        - interval from previous note
        
        Args:
            solo_data (pd.DataFrame): DataFrame containing melody, beats, and solo_info DataFrames
        
        Returns:
            Solo Data with melodic features added
        """
        solo_df = solo_df.copy()
        
        # Calculate interval from previous note
        solo_df['prev_interval'] = solo_df['pitch'].diff().fillna(0).astype(int)     

        return solo_df

    def _add_harmonic_features(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add harmonic-related features to the solo data.
        - key center
        - chord-relative pitch
        - original key shift

        Args:
            solo_data (pd.DataFrame): DataFrame containing melody, beats, and solo_info DataFrames
        
        Returns:
            Solo Data with harmonic features added
        """
        solo_df = solo_df.copy()

        # Add key center (parse the key from solo_info)
        key_value = solo_df['key'].iloc[0] if len(solo_df) > 0 else 'C'
        parsed_key = self.chord_parser.parse_chord(key_value)
        solo_df['key_center'] = parsed_key['root'] if parsed_key else 0

        # Add original key shift (0)
        solo_df['key_shift'] = 0

        # Calculate chord-relative pitch
        solo_df['chord_rel_pitch'] = solo_df.apply(
            lambda row: (row['pitch'] - row['chord_root']) % 12 
            if pd.notna(row['chord_root']) else 0,
            axis=1
        )
        
        return solo_df

    def _calculate_position_grid(self, beat: float, tatum: float, division: int) -> int:
        """
        Calculate position in 48-grid system using WJD's structural rhythm info.
        
        Formula: Pos = (beat - 1) * 12 + Round((tatum - 1) / division * 12)
        
        Args:
            beat: Beat number in bar (1-4 for 4/4 time)
            tatum: Tatum number within beat (1-indexed)
            division: Division of the beat
        
        Returns:
            Position in 48-grid (0-47)
        """

        beat = safe_int(beat, 1)
        tatum = safe_int(tatum, 1)
        division = safe_int(division, 1)
        
        # Avoid division by zero
        if division == 0:
            division = 1
        
        # Calculate position
        pos = (beat - 1) * GRID_PER_BEAT + round((tatum - 1) / division * GRID_PER_BEAT)
        
        # Clip to valid range
        pos = int(clip_to_range(pos, 0, GRID_PER_BAR - 1))
        
        return pos

    def _calculate_duration_grid(self, duration_sec: float, beatdur_sec: float) -> int:
        """
        Convert duration from seconds to 12-grid units.
        
        Args:
            duration_sec: Duration in seconds
            beatdur_sec: Duration of one beat in seconds
        
        Returns:
            Duration in grid units (clipped to max 48)
        """
        if pd.isna(duration_sec) or pd.isna(beatdur_sec) or beatdur_sec == 0:
            return 1  # Default to minimum duration
        
        # Convert to grid units
        dur_grid = round(safe_divide(duration_sec, beatdur_sec, 1.0) * GRID_PER_BEAT)
        
        # Clip to valid range (minimum 1, maximum 48)
        dur_grid = int(clip_to_range(dur_grid, 1, GRID_PER_BAR))
        
        return dur_grid
    
    def _select_final_features(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the columns needed for training.
        
        Args:
            solo_df (pd.DataFrame): DataFrame with all features
        
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        feature_cols = [
            # Metadata
            'eventid', 'melid', 'beatid',
            # Position info
            'bar', 'beat', 'chorus_id',
            # Raw features
            'pitch', 'onset', 'duration',
            # Derived rhythmic features
            'pos_grid', 'dur_grid',
            # Derived melodic features
            'prev_interval',
            # Derived harmonic features
            'chord', 'chord_root', 'chord_quality',
            'next_chord', 'next_chord_root', 'next_chord_quality',
            'chord_rel_pitch', 'key_center', 'key_shift',
            # Context
            'key', 'avgtempo', 'performer', 'style', 'title', 'signature'
        ]
        
        # Only select columns that exist
        existing_cols = [col for col in feature_cols if col in solo_df.columns]
        return solo_df[existing_cols].copy()
    
    def _convert_dtypes(self, solo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all columns to proper data types.
        
        Data type strategy:
        - IDs and categorical integers: Int64 (nullable integer)
        - Continuous numeric values: float64
        - Text fields: string
        
        Args:
            solo_df (pd.DataFrame): DataFrame with features
        
        Returns:
            pd.DataFrame: DataFrame with proper dtypes
        """
        solo_df = solo_df.copy()
        
        # Integer columns (nullable - can have NA values)
        int_cols = [
            'melid', 'eventid', 'beatid',           # IDs
            'bar', 'beat', 'chorus_id',             # Position info
            'pitch',                                 # MIDI pitch (can be NA for rests)
            'pos_grid', 'dur_grid',                 # Rhythmic grid values
            'prev_interval',                         # Melodic interval
            'chord_root', 'chord_quality',          # Chord info
            'next_chord_root', 'next_chord_quality', # Next chord info
            'chord_rel_pitch',                       # Harmonic features
            'key_center', 'key_shift'               # Key info
        ]
        
        for col in int_cols:
            if col in solo_df.columns:
                solo_df[col] = solo_df[col].astype(pd.Int64Dtype())
        
        # Float columns (continuous values)
        float_cols = [
            'onset', 'duration',  # Time values in seconds
            'avgtempo',           # Tempo in BPM
            'beatdur'             # Beat duration in seconds
        ]
        
        for col in float_cols:
            if col in solo_df.columns:
                solo_df[col] = solo_df[col].astype('float64')
        
        # String columns (text data)
        string_cols = [
            'chord', 'next_chord',  # Chord symbols
            'key',                   # Key signature
            'performer', 'style', 'title',  # Metadata
            'signature'              # Time signature
        ]
        
        for col in string_cols:
            if col in solo_df.columns:
                # Convert to string type (pandas string dtype, not object)
                solo_df[col] = solo_df[col].astype('string')
        
        return solo_df

        
if __name__ == "__main__":
    # Check if Database exists
    if not Path(DB_PATH).exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        exit(1)

    # Run preprocessing
    preprocessor = Preprocessor(db_path=DB_PATH)
    processed_df = preprocessor.process()

    # Save output
    processed_df.to_pickle(OUTPUT_PKL)
    processed_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Processed {len(processed_df)} rows")
    print(f"   - Saved to {OUTPUT_PKL}")
    print(f"   - Saved to {OUTPUT_CSV}")
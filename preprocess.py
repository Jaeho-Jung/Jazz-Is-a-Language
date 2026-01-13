"""
WJD Bebop/Hardbop Solo Preprocessing Module

This module extracts and preprocesses jazz solo data from the Weimar Jazz Database (WJD)
for training HMM and Deep Learning models (RNN/Transformer).

Based on requirements:
- requirement-preprocess.md: Functional requirements
- model_data_logic.md: Event-based sequence modeling rationale
- data-format.md: SQLite3 database schema
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path


# ============================================================================
# CONSTANTS
# ============================================================================

# Database path
DB_PATH = "data/wjazzd.db"

# Output paths
OUTPUT_PKL = "data/wjd_bebop_preprocessed.pkl"
OUTPUT_CSV = "data/wjd_bebop_preprocessed.csv"

# Filtering criteria
TARGET_STYLES = ['BEBOP', 'HARDBOP']
TARGET_SIGNATURE = '4/4'
TARGET_MELODY_TYPE = 'SOLO'

# Data augmentation: Balanced 12-key transposition
# Range: -5 to +6 semitones (total 12 keys including original)
AUGMENTATION_SHIFTS = list(range(-5, 7))  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

# MIDI pitch range
MIDI_MIN = 0
MIDI_MAX = 127

# Rhythm grid constants
GRID_PER_BEAT = 12  # 1 beat = 12 grid units (supports 16th notes and triplets)
GRID_PER_BAR = 48   # 4/4 time signature: 4 beats * 12 = 48

# Chord quality mapping
CHORD_QUALITY_MAP = {
    'Maj': ['maj', 'maj7', 'maj6', 'maj9', '6', '^', 'M7', 'M'],
    'min': ['min', 'min7', 'min6', 'min9', 'm7', 'm6', 'm9', 'm', '-'],
    'dom': ['7', '9', '11', '13', 'dom', 'alt', 'b9', '#9', 'b13', '#11'],
    'half-dim': ['m7b5', 'ø', 'half-dim'],
    'dim': ['dim', 'o', 'dim7', 'o7'],
    'sus': ['sus', 'sus4', 'sus2']
}

# Note names for root extraction
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
        return 5  # Default to 'sus/other'
    
    chord_str = str(chord_str).strip().lower()
    
    # Remove root note
    if len(chord_str) >= 2 and chord_str[1] in ['#', 'b']:
        quality_str = chord_str[2:]
    elif len(chord_str) >= 1:
        quality_str = chord_str[1:]
    else:
        return 5
    
    # Check each quality category
    for quality_idx, (quality_name, patterns) in enumerate(CHORD_QUALITY_MAP.items()):
        for pattern in patterns:
            if pattern.lower() in quality_str:
                return quality_idx
    
    return 5  # Default to 'sus/other'


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
    return dominant_root, 2  # 2 = dom quality


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


def calculate_position_grid(beat: int, tatum: int, division: int) -> int:
    """
    Calculate position in 48-grid system using WJD's structural rhythm info.
    
    This robust method avoids onset-based calculation errors due to micro-timing.
    
    Formula: Pos = (beat - 1) * 12 + Round((tatum - 1) / division * 12)
    
    Args:
        beat: Beat number in bar (1-4 for 4/4 time)
        tatum: Tatum number within beat (1-indexed)
        division: Division of the beat
    
    Returns:
        Position in 48-grid (0-47)
    """
    if pd.isna(beat) or pd.isna(tatum) or pd.isna(division):
        return 0
    
    beat = int(beat)
    tatum = int(tatum)
    division = int(division)
    
    # Avoid division by zero
    if division == 0:
        division = 1
    
    # Calculate position
    pos = (beat - 1) * GRID_PER_BEAT + round((tatum - 1) / division * GRID_PER_BEAT)
    
    # Clip to valid range
    pos = max(0, min(GRID_PER_BAR - 1, pos))
    
    return pos


def calculate_duration_grid(duration_sec: float, beat_dur_sec: float) -> int:
    """
    Convert duration from seconds to 12-grid units.
    
    Args:
        duration_sec: Duration in seconds
        beat_dur_sec: Duration of one beat in seconds
    
    Returns:
        Duration in grid units (clipped to max 48)
    """
    if pd.isna(duration_sec) or pd.isna(beat_dur_sec) or beat_dur_sec == 0:
        return 1  # Default to minimum duration
    
    # Convert to grid units
    dur_grid = round((duration_sec / beat_dur_sec) * GRID_PER_BEAT)
    
    # Clip to valid range (minimum 1, maximum 48)
    dur_grid = max(1, min(GRID_PER_BAR, dur_grid))
    
    return dur_grid


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def get_target_melids(db_path: str) -> List[int]:
    """
    Extract melids that match the filtering criteria:
    - melody_type.type = 'SOLO'
    - solo_info.style in ['BEBOP', 'HARDBOP']
    - beats.signature = '4/4'
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        List of melids
    """
    conn = sqlite3.connect(db_path)
    
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


def load_solo_data(db_path: str, melid: int) -> Dict[str, pd.DataFrame]:
    """
    Load all relevant data for a single solo.
    
    Args:
        db_path: Path to SQLite database
        melid: Melody ID
    
    Returns:
        Dictionary containing DataFrames for melody, beats, and solo_info
    """
    conn = sqlite3.connect(db_path)
    
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


# ============================================================================
# DATA PROCESSING
# ============================================================================

def align_chords_to_melody(melody_df: pd.DataFrame, beats_df: pd.DataFrame) -> pd.DataFrame:
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
    beats_df['chord_root'] = beats_df['chord'].apply(parse_chord_root)
    beats_df['chord_quality'] = beats_df['chord'].apply(parse_chord_quality)
    
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


def handle_pickup_measures(melody_df: pd.DataFrame, beats_df: pd.DataFrame) -> pd.DataFrame:
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
        dominant_root, dominant_quality = calculate_secondary_dominant(int(first_chord_root))
        
        # Fill pickup measures
        for idx in range(first_valid_idx):
            if pd.isna(melody_df.at[idx, 'chord_root']):
                melody_df.at[idx, 'chord_root'] = dominant_root
                melody_df.at[idx, 'chord_quality'] = dominant_quality
                melody_df.at[idx, 'chord'] = f"{NOTE_NAMES[dominant_root]}7"
    
    return melody_df


def process_solo(solo_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Process a single solo: align chords, calculate features, etc.
    
    Args:
        solo_data: Dictionary with melody, beats, and solo_info DataFrames
    
    Returns:
        Processed DataFrame with all features
    """
    melody_df = solo_data['melody'].copy()
    beats_df = solo_data['beats'].copy()
    solo_info = solo_data['solo_info'].iloc[0]
    
    # Get key center
    key_center = key_to_pitch_class(solo_info['key'])
    
    # Align chords to melody
    melody_df = align_chords_to_melody(melody_df, beats_df)
    
    # Handle pickup measures
    melody_df = handle_pickup_measures(melody_df, beats_df)
    
    # Normalize pitch to nearest integer
    melody_df['pitch_normalized'] = melody_df['pitch'].round().astype(int)
    
    # Calculate position grid
    melody_df['pos_grid'] = melody_df.apply(
        lambda row: calculate_position_grid(row['beat'], row['tatum'], row['division']),
        axis=1
    )
    
    # Calculate duration grid
    melody_df['dur_grid'] = melody_df.apply(
        lambda row: calculate_duration_grid(row['duration'], row['beatdur']),
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


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_solo(processed_df: pd.DataFrame, shift: int) -> pd.DataFrame:
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


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_training_dataset(processed_solos: List[pd.DataFrame]) -> pd.DataFrame:
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


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_wjd(db_path: str, output_pkl: str, output_csv: str):
    """
    Main preprocessing pipeline.
    
    Args:
        db_path: Path to WJD SQLite database
        output_pkl: Output pickle file path
        output_csv: Output CSV file path
    """
    print("=" * 80)
    print("WJD Bebop/Hardbop Solo Preprocessing Pipeline")
    print("=" * 80)
    
    # Step 1: Extract target melids
    print("\n[Step 1] Extracting target melids...")
    melids = get_target_melids(db_path)
    
    if len(melids) == 0:
        print("No solos found matching the criteria!")
        return
    
    # Step 2: Process each solo
    print(f"\n[Step 2] Processing {len(melids)} solos...")
    processed_solos = []
    
    for i, melid in enumerate(melids):
        print(f"  Processing melid {melid} ({i+1}/{len(melids)})...")
        
        try:
            # Load solo data
            solo_data = load_solo_data(db_path, melid)
            
            # Process solo
            processed_df = process_solo(solo_data)
            
            # Add original key shift (0)
            processed_df['key_shift'] = 0
            
            processed_solos.append(processed_df)
            
        except Exception as e:
            print(f"    Error processing melid {melid}: {e}")
            continue
    
    print(f"  Successfully processed {len(processed_solos)} solos.")
    
    # Step 3: Data augmentation
    print(f"\n[Step 3] Augmenting data with {len(AUGMENTATION_SHIFTS)} transpositions...")
    augmented_solos = []
    
    for solo_df in processed_solos:
        for shift in AUGMENTATION_SHIFTS:
            if shift == 0:
                # Original already added
                augmented_solos.append(solo_df)
            else:
                aug_df = augment_solo(solo_df, shift)
                augmented_solos.append(aug_df)
    
    print(f"  Total augmented solos: {len(augmented_solos)}")
    
    # Step 4: Create training dataset
    print("\n[Step 4] Creating training dataset...")
    final_df = create_training_dataset(augmented_solos)
    
    print(f"  Total training examples: {len(final_df)}")
    print(f"  Features: {list(final_df.columns)}")
    
    # Step 5: Save output
    print("\n[Step 5] Saving output files...")
    
    # Save as pickle
    final_df.to_pickle(output_pkl)
    print(f"  Saved: {output_pkl}")
    
    # Save as CSV
    final_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Original solos: {len(melids)}")
    print(f"Augmentation shifts: {AUGMENTATION_SHIFTS}")
    print(f"Total augmented solos: {len(augmented_solos)}")
    print(f"Total training examples: {len(final_df)}")
    print(f"\nDataset shape: {final_df.shape}")
    print(f"\nSample statistics:")
    print(f"  Pitch range: {final_df['target_pitch'].min()} - {final_df['target_pitch'].max()}")
    print(f"  Duration range: {final_df['target_dur'].min()} - {final_df['target_dur'].max()}")
    print(f"  Position range: {final_df['pos_grid'].min()} - {final_df['pos_grid'].max()}")
    print(f"  Chord qualities: {sorted(final_df['chord_quality'].unique())}")
    print(f"  Styles: {final_df['style'].unique()}")
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Please ensure the WJD database is in the correct location.")
        exit(1)
    
    # Run preprocessing
    preprocess_wjd(DB_PATH, OUTPUT_PKL, OUTPUT_CSV)

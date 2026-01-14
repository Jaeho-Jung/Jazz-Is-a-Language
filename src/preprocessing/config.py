"""
Configuration and Constants for WJD Preprocessing

Centralizes all configuration parameters, constants, and mappings.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Database path
DB_PATH = "data/wjazzd.db"

# Output paths
OUTPUT_PKL = "data/wjd_bebop_preprocessed.pkl"
OUTPUT_CSV = "data/wjd_bebop_preprocessed.csv"

# ============================================================================
# FILTERING CRITERIA
# ============================================================================

TARGET_STYLES = ['BEBOP', 'HARDBOP']
TARGET_SIGNATURE = '4/4'
TARGET_MELODY_TYPE = 'SOLO'

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

# Balanced 12-key transposition: -5 to +6 semitones (total 12 keys)
AUGMENTATION_SHIFTS = list(range(-5, 7))  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

# ============================================================================
# MIDI CONSTANTS
# ============================================================================

MIDI_MIN = 0
MIDI_MAX = 127

# ============================================================================
# RHYTHM GRID CONSTANTS
# ============================================================================

GRID_PER_BEAT = 12  # 1 beat = 12 grid units (supports 16th notes and triplets)
GRID_PER_BAR = 48   # 4/4 time signature: 4 beats * 12 = 48

# ============================================================================
# CHORD QUALITY MAPPING
# ============================================================================

CHORD_QUALITY_MAP = {
    'Maj': ['maj', 'maj7', 'maj6', 'maj9', '6', '^', 'M7', 'M'],
    'min': ['min', 'min7', 'min6', 'min9', 'm7', 'm6', 'm9', 'm', '-'],
    'dom': ['7', '9', '11', '13', 'dom', 'alt', 'b9', '#9', 'b13', '#11'],
    'half-dim': ['m7b5', 'ø', 'half-dim'],
    'dim': ['dim', 'o', 'dim7', 'o7'],
    'sus': ['sus', 'sus4', 'sus2']
}

# Quality indices
QUALITY_MAJ = 0
QUALITY_MIN = 1
QUALITY_DOM = 2
QUALITY_HALF_DIM = 3
QUALITY_DIM = 4
QUALITY_SUS = 5

# ============================================================================
# NOTE NAMES
# ============================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

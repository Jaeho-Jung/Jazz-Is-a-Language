"""
Configuration and Constants for WJD Proprocessor
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Database path
DB_PATH = 'data/wjazzd.db'

# Output paths
OUTPUT_PKL = 'data/wjd_bebop_preprocessed.pkl'
OUTPUT_CSV = 'data/wjd_bebop_preprocessed.csv'


# ============================================================================
# FILTERING CRITERIA
# ============================================================================


TARGET_MELODY_TYPE = 'SOLO'
TARGET_STYLES = ['BEBOP', 'HARDBOP']
TARGET_SIGNATURE = '4/4'

# ============================================================================
# NOTE NAMES
# ============================================================================

NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ============================================================================
# RHYTHM GRID CONSTANTS
# ============================================================================

GRID_PER_BEAT = 12  # 1 beat = 12 grid units (supports 16th notes and triplets)
GRID_PER_BAR = 48   # 4/4 time signature: 4 beats * 12 = 48

# ============================================================================
# CHORD QUALITY MAPPINGS
# ============================================================================

CHORD_QUALITY_MAP = {
    'Maj': ['j7', '6', '69'],
    'min': ['-', '-7', '-6'],
    'dom': ['7', '79b', '7913'],
    'half-dim': ['m7b5'],
}

"""
Configuration for NumPy LSTM Jazz Solo Generator
"""

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_PATH = 'data/wjd_bebop_preprocessed.pkl'

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 64
SEQ_LEN = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Optimizer settings
MOMENTUM = 0.9
RMSPROP_DECAY = 0.99
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.01

# ============================================================================
# LSTM ARCHITECTURE
# ============================================================================
LSTM_HIDDEN_SIZE = 256

# ============================================================================
# FEATURE DIMENSIONS (VOCAB SIZES)
# ============================================================================
# Pitch
VOCAB_SIZE_PITCH = 129          # 0-127 (MIDI), 128 (Rest)
VOCAB_SIZE_REL_PITCH = 13       # 0-11 (pitch class), 12 (Rest/NC)

# Rhythm
VOCAB_SIZE_GRID_POS = 48        # 0-47 (beat positions)
# Duration vocab size is dynamic, determined from dataset

# Harmony
VOCAB_SIZE_CHORD_ROOT = 13      # 0-11 (pitch class), 12 (NC)
VOCAB_SIZE_CHORD_ROOT_REL = 13  # 0-11 (relative to current note), 12 (NC)
VOCAB_SIZE_CHORD_QUALITY = 7    # 6 types + NC

# Melodic
VOCAB_SIZE_PREV_INTERVAL = 25   # -12 to +12, offset by 12 to get 0-24

# ============================================================================
# EMBEDDING SIZES
# ============================================================================
EMBED_SIZE_PITCH = 32
EMBED_SIZE_REL_PITCH = 8
EMBED_SIZE_DURATION = 8
EMBED_SIZE_GRID_POS = 8
EMBED_SIZE_CHORD_ROOT = 8
EMBED_SIZE_CHORD_ROOT_REL = 8
EMBED_SIZE_CHORD_QUALITY = 4
EMBED_SIZE_PREV_INTERVAL = 8

# ============================================================================
# COMPUTED CONSTANTS
# ============================================================================
TOTAL_EMBED_SIZE = (
    EMBED_SIZE_PITCH +
    EMBED_SIZE_REL_PITCH +
    EMBED_SIZE_DURATION +
    EMBED_SIZE_GRID_POS +
    EMBED_SIZE_CHORD_ROOT +
    EMBED_SIZE_CHORD_ROOT_REL +
    EMBED_SIZE_CHORD_QUALITY +
    EMBED_SIZE_CHORD_ROOT +         # next_chord_root
    EMBED_SIZE_CHORD_ROOT_REL +     # next_chord_root_rel
    EMBED_SIZE_CHORD_QUALITY +      # next_chord_quality
    EMBED_SIZE_PREV_INTERVAL
)
# = 32 + 8 + 8 + 8 + 8 + 8 + 4 + 8 + 8 + 4 + 8 = 104

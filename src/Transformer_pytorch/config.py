"""
Configuration for Transformer Jazz Solo Generator
"""

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_PATH = 'data/wjd_bebop_preprocessed.pkl'

# ============================================================================
# TRANSFORMER HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 256
D_MODEL = 192               # Model dimension (must be divisible by N_HEADS)
N_HEADS = 6                 # Number of attention heads
NUM_LAYERS = 4              # Number of encoder layers
DROPOUT = 0.3               # Stronger regularization to delay overfitting
SEQ_LEN = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
WEIGHT_DECAY = 0.05     # Increased from 0.01
LABEL_SMOOTHING = 0.1   # Softens targets to prevent overconfident predictions

# ============================================================================
# FEATURE DIMENSIONS (VOCAB SIZES)
# ============================================================================
# Pitch
VOCAB_SIZE_PITCH = 129      # 0-127 (MIDI), 128 (Rest)
VOCAB_SIZE_REL_PITCH = 13   # 0-11 (pc), 12 (Rest/Pad)

# Rhythm
VOCAB_SIZE_GRID_POS = 48    # 0-47
VOCAB_SIZE_METRIC_POS = 48  # Alias for GRID_POS
# Duration vocab size is dynamic, determined from dataset

# Harmony
VOCAB_SIZE_CHORD_ROOT = 13      # 0-11, 12 (NC)
VOCAB_SIZE_CHORD_ROOT_REL = 13  # 0-11, 12 (NC)
VOCAB_SIZE_CHORD_QUALITY = 7    # 6 types + NC

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

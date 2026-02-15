"""
Configuration for RNN Jazz Solo Generator

Optimized for 784k sample dataset with simplified 7-feature architecture.
"""

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_PATH = 'data/wjd_bebop_preprocessed.pkl'

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
SEQ_LEN = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01

# ============================================================================
# FEATURE DIMENSIONS (VOCAB SIZES)
# ============================================================================
VOCAB_SIZE_PITCH = 129          # 0-127 (MIDI), 128 (Rest)
VOCAB_SIZE_REL_PITCH = 13       # 0-11 (pitch class), 12 (Rest)
VOCAB_SIZE_DURATION = 32        # Dynamic, but reserve size
VOCAB_SIZE_PREV_INTERVAL = 25   # -12 to +12 mapped to 0-24
VOCAB_SIZE_CHORD_ROOT = 13      # 0-11, 12 (NC)
VOCAB_SIZE_CHORD_QUALITY = 7    # 6 types + NC
VOCAB_SIZE_METRIC_POS = 48      # 0-47 (48th note grid position)

# ============================================================================
# EMBEDDING SIZES (Optimized for 784k sample dataset)
# ============================================================================
EMBED_SIZE_PITCH = 48           # Primary signal, high capacity
EMBED_SIZE_REL_PITCH = 16       # Strong harmonic prior
EMBED_SIZE_DURATION = 16        # Rhythmic identity
EMBED_SIZE_PREV_INTERVAL = 16   # Melodic contour
EMBED_SIZE_CHORD_ROOT = 16      # Harmonic context
EMBED_SIZE_CHORD_QUALITY = 8    # Low cardinality
EMBED_SIZE_METRIC_POS = 16      # Beat position

# Total input size: 48 + 16 + 16 + 16 + 16 + 8 + 16 = 136

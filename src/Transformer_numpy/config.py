# Data
DATA_PATH = 'data/wjd_jazz_preprocessed.pkl'

# Training (match Transformer_pytorch)
BATCH_SIZE = 512
SEQ_LEN = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
LABEL_SMOOTHING = 0.1

# Optimizer
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9
RMSPROP_DECAY = 0.99

# Architecture (match Transformer_pytorch)
TRANSFORMER_HIDDEN_SIZE = 128   # = D_MODEL
NUM_HEADS = 4                   # = N_HEADS
NUM_BLOCKS = 3                  # = NUM_LAYERS
DROPOUT_RATE = 0.3              # = DROPOUT
MAX_SEQ_LEN = SEQ_LEN

# Vocab sizes — 7 features, same as Transformer_pytorch
VOCAB_SIZE_PITCH = 129          # 0-127 MIDI + 128 rest
VOCAB_SIZE_REL_PITCH = 13       # 0-11 + 12 rest
VOCAB_SIZE_GRID_POS = 48        # metric_pos: 0-47
VOCAB_SIZE_CHORD_ROOT = 13      # 0-11 + 12 NC
VOCAB_SIZE_CHORD_QUALITY = 7    # 6 types + NC
VOCAB_SIZE_PREV_INTERVAL = 25   # -12..+12 → 0..24

# Embed sizes — match Transformer_pytorch
EMBED_SIZE_PITCH = 32
EMBED_SIZE_REL_PITCH = 8
EMBED_SIZE_DURATION = 8
EMBED_SIZE_GRID_POS = 8
EMBED_SIZE_CHORD_ROOT = 8
EMBED_SIZE_CHORD_QUALITY = 4
EMBED_SIZE_PREV_INTERVAL = 8

# Total: 32+8+8+8+8+4+8 = 76
TOTAL_EMBED_SIZE = 76

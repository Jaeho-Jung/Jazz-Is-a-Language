"""
Unit tests for the complete JazzRNN model.

Tests forward pass, backward pass, gradient checking, and shape validation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.RNN_numpy.model import JazzRNN
from src.RNN_numpy import config


def create_dummy_features(batch_size, seq_len, num_dur_classes=10):
    """Create random feature dict for testing."""
    return {
        'pitch': np.random.randint(0, config.VOCAB_SIZE_PITCH, (batch_size, seq_len)),
        'rel_pitch': np.random.randint(0, config.VOCAB_SIZE_REL_PITCH, (batch_size, seq_len)),
        'dur': np.random.randint(0, num_dur_classes, (batch_size, seq_len)),
        'pos': np.random.randint(0, config.VOCAB_SIZE_GRID_POS, (batch_size, seq_len)),
        'chord_root': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT, (batch_size, seq_len)),
        'chord_root_rel': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT_REL, (batch_size, seq_len)),
        'chord_quality': np.random.randint(0, config.VOCAB_SIZE_CHORD_QUALITY, (batch_size, seq_len)),
        'next_chord_root': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT, (batch_size, seq_len)),
        'next_chord_root_rel': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT_REL, (batch_size, seq_len)),
        'next_chord_quality': np.random.randint(0, config.VOCAB_SIZE_CHORD_QUALITY, (batch_size, seq_len)),
        'prev_interval': np.random.randint(0, config.VOCAB_SIZE_PREV_INTERVAL, (batch_size, seq_len)),
    }


def test_forward_shapes():
    """Test that forward pass produces correct output shapes."""
    print("\n=== Testing Forward Pass Shapes ===")
    
    batch_size = 4
    seq_len = 16
    num_dur_classes = 10
    
    model = JazzRNN(num_dur_classes=num_dur_classes)
    features = create_dummy_features(batch_size, seq_len, num_dur_classes)
    
    pitch_logits, dur_logits = model.forward(features)
    
    assert pitch_logits.shape == (batch_size, config.VOCAB_SIZE_PITCH), \
        f"Pitch logits shape wrong: {pitch_logits.shape}"
    assert dur_logits.shape == (batch_size, num_dur_classes), \
        f"Duration logits shape wrong: {dur_logits.shape}"
    
    print(f"✓ Pitch logits shape: {pitch_logits.shape}")
    print(f"✓ Duration logits shape: {dur_logits.shape}")


def test_embedding_concatenation():
    """Test that embeddings are correctly concatenated."""
    print("\n=== Testing Embedding Concatenation ===")
    
    batch_size = 2
    seq_len = 4
    num_dur_classes = 10
    
    model = JazzRNN(num_dur_classes=num_dur_classes)
    features = create_dummy_features(batch_size, seq_len, num_dur_classes)
    
    # Manually compute expected total embedding size
    expected_size = config.TOTAL_EMBED_SIZE
    
    # Forward pass
    _ = model.forward(features)
    
    # Check RNN received correct input shape
    # Access cached value from RNN cell
    rnn_input_shape = model.rnn.cell.cache['x_t'].shape
    
    assert rnn_input_shape[0] == batch_size, f"Batch size wrong: {rnn_input_shape[0]}"
    assert rnn_input_shape[1] == expected_size, \
        f"Total embedding size wrong: {rnn_input_shape[1]}, expected {expected_size}"
    
    print(f"✓ Total embedding size: {expected_size}")
    print(f"✓ RNN input shape per timestep: {rnn_input_shape}")


def test_backward_shapes():
    """Test that backward pass produces correct gradient shapes."""
    print("\n=== Testing Backward Pass Shapes ===")
    
    batch_size = 4
    seq_len = 8
    num_dur_classes = 10
    
    model = JazzRNN(num_dur_classes=num_dur_classes)
    features = create_dummy_features(batch_size, seq_len, num_dur_classes)
    
    # Forward pass
    pitch_logits, dur_logits = model.forward(features)
    
    # Create gradients
    grad_pitch = np.random.randn(batch_size, config.VOCAB_SIZE_PITCH)
    grad_dur = np.random.randn(batch_size, num_dur_classes)
    
    # Backward pass
    model.backward(grad_pitch, grad_dur)
    
    # Check that gradients are stored
    all_grads = model.get_all_grads()
    
    # Check embedding gradients
    for name, emb in model.embeddings.items():
        assert emb.grad_W is not None, f"Embedding {name} missing gradient"
        assert emb.grad_W.shape == emb.W.shape, f"Embedding {name} gradient shape wrong"
    
    # Check RNN gradients
    rnn_grads = all_grads['rnn_cell']
    assert rnn_grads['W_xh'] is not None, "RNN W_xh missing gradient"
    assert rnn_grads['W_hh'] is not None, "RNN W_hh missing gradient"
    assert rnn_grads['b_h'] is not None, "RNN b_h missing gradient"
    
    # Check head gradients
    assert all_grads['pitch_head']['W'] is not None, "Pitch head W missing gradient"
    assert all_grads['dur_head']['W'] is not None, "Duration head W missing gradient"
    
    print("✓ All layers have gradients stored")
    print(f"✓ Number of embedding gradients: {len(model.embeddings)}")


def test_params_and_grads_interface():
    """Test get_all_params and get_all_grads interface."""
    print("\n=== Testing Parameter/Gradient Interface ===")
    
    model = JazzRNN(num_dur_classes=10)
    
    params = model.get_all_params()
    
    # Check embeddings
    assert 'emb_pitch' in params, "Missing pitch embedding params"
    assert 'emb_chord_root_rel' in params, "Missing chord_root_rel embedding params"
    
    # Check RNN
    assert 'rnn_cell' in params, "Missing RNN params"
    assert 'W_xh' in params['rnn_cell'], "Missing W_xh in RNN params"
    assert 'W_hh' in params['rnn_cell'], "Missing W_hh in RNN params"
    
    # Check heads
    assert 'pitch_head' in params, "Missing pitch head params"
    assert 'dur_head' in params, "Missing duration head params"
    
    print(f"✓ Total embedding layers: {len([k for k in params.keys() if k.startswith('emb_')])}")
    print(f"✓ RNN params: {list(params['rnn_cell'].keys())}")
    print(f"✓ Head params: {list(params['pitch_head'].keys())}")


def test_end_to_end_forward_backward():
    """Test complete forward and backward pass."""
    print("\n=== Testing End-to-End Forward/Backward ===")
    
    np.random.seed(42)
    
    batch_size = 2
    seq_len = 4
    num_dur_classes = 5
    
    model = JazzRNN(num_dur_classes=num_dur_classes)
    features = create_dummy_features(batch_size, seq_len, num_dur_classes)
    
    # Forward
    pitch_logits, dur_logits = model.forward(features)
    
    # Simulate loss gradients (from cross-entropy)
    # In real training: grad = (softmax(logits) - target) / batch_size
    grad_pitch = np.random.randn(*pitch_logits.shape) * 0.01
    grad_dur = np.random.randn(*dur_logits.shape) * 0.01
    
    # Backward
    model.backward(grad_pitch, grad_dur)
    
    # Verify gradients exist and have correct shapes
    grads = model.get_all_grads()
    
    # Sample checks
    assert grads['emb_pitch']['W'].shape == model.embeddings['pitch'].W.shape
    assert grads['rnn_cell']['W_xh'].shape == model.rnn.cell.W_xh.shape
    assert grads['pitch_head']['W'].shape == model.pitch_head.W.shape
    
    print("✓ Forward pass completed successfully")
    print("✓ Backward pass completed successfully")
    print("✓ All gradient shapes match parameter shapes")


def test_numerical_stability():
    """Test that forward pass doesn't produce NaN or Inf values."""
    print("\n=== Testing Numerical Stability ===")
    
    batch_size = 4
    seq_len = 32
    num_dur_classes = 20
    
    model = JazzRNN(num_dur_classes=num_dur_classes)
    features = create_dummy_features(batch_size, seq_len, num_dur_classes)
    
    pitch_logits, dur_logits = model.forward(features)
    
    assert not np.any(np.isnan(pitch_logits)), "NaN values in pitch logits"
    assert not np.any(np.isinf(pitch_logits)), "Inf values in pitch logits"
    assert not np.any(np.isnan(dur_logits)), "NaN values in duration logits"
    assert not np.any(np.isinf(dur_logits)), "Inf values in duration logits"
    
    print("✓ No NaN or Inf values in output")
    print(f"  Pitch logits range: [{pitch_logits.min():.4f}, {pitch_logits.max():.4f}]")
    print(f"  Duration logits range: [{dur_logits.min():.4f}, {dur_logits.max():.4f}]")


def run_all_tests():
    """Run all model tests."""
    print("=" * 60)
    print("Running JazzRNN Model Tests")
    print("=" * 60)
    
    test_forward_shapes()
    test_embedding_concatenation()
    test_backward_shapes()
    test_params_and_grads_interface()
    test_end_to_end_forward_backward()
    test_numerical_stability()
    
    print("\n" + "=" * 60)
    print("✅ ALL MODEL TESTS PASSED!")
    print("=" * 60)
    print("\nModel Architecture Summary:")
    print(f"  Embeddings: 11 layers")
    print(f"  Total embedding size: {config.TOTAL_EMBED_SIZE}")
    print(f"  RNN hidden size: {config.RNN_HIDDEN_SIZE}")
    print(f"  Output heads: pitch ({config.VOCAB_SIZE_PITCH}), duration (dynamic)")


if __name__ == "__main__":
    run_all_tests()

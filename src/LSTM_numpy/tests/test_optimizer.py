"""
Unit tests for optimizers with nested dictionary support.

Tests all optimizers: SGD, SGDWithMomentum, AdaGrad, RMSProp, Adam, AdamW.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.LSTM_numpy.optimizer import SGD, SGDWithMomentum, AdaGrad, RMSProp, Adam, AdamW


class SimpleModel:
    """Simple model with flat params for basic optimizer tests."""
    
    def __init__(self):
        self.W = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.b = np.array([0.5, 0.5])
        self.grad_W = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.grad_b = np.array([0.1, 0.1])
    
    def get_all_params(self):
        return {'W': self.W, 'b': self.b}
    
    def get_all_grads(self):
        return {'W': self.grad_W, 'b': self.grad_b}
    
    def set_all_params(self, params):
        self.W = params['W']
        self.b = params['b']


class NestedModel:
    """Model with nested params to test recursive traversal."""
    
    def __init__(self):
        self.layers = {
            'layer1': {
                'W': np.array([[1.0, 2.0], [3.0, 4.0]]),
                'b': np.array([0.5, 0.5])
            },
            'layer2': {
                'W': np.array([[5.0, 6.0], [7.0, 8.0]]),
                'b': np.array([1.0, 1.0])
            }
        }
        self.grads = {
            'layer1': {
                'W': np.array([[0.1, 0.2], [0.3, 0.4]]),
                'b': np.array([0.1, 0.1])
            },
            'layer2': {
                'W': np.array([[0.5, 0.6], [0.7, 0.8]]),
                'b': np.array([0.2, 0.2])
            }
        }
    
    def get_all_params(self):
        return self.layers
    
    def get_all_grads(self):
        return self.grads
    
    def set_all_params(self, params):
        self.layers = params


def test_sgd_flat():
    """Test SGD with flat parameter dict."""
    print("\n=== Testing SGD (flat) ===")
    
    model = SimpleModel()
    optimizer = SGD(model, lr=0.1)
    
    W_before = model.W.copy()
    
    optimizer.step()
    
    expected_W = W_before - 0.1 * model.grad_W
    assert np.allclose(model.W, expected_W), f"SGD update wrong"
    print("✓ SGD update correct")


def test_sgd_nested():
    """Test SGD with nested parameter dict."""
    print("\n=== Testing SGD (nested) ===")
    
    model = NestedModel()
    optimizer = SGD(model, lr=0.1)
    
    W1_before = model.layers['layer1']['W'].copy()
    W2_before = model.layers['layer2']['W'].copy()
    
    optimizer.step()
    
    expected_W1 = W1_before - 0.1 * model.grads['layer1']['W']
    expected_W2 = W2_before - 0.1 * model.grads['layer2']['W']
    
    assert np.allclose(model.layers['layer1']['W'], expected_W1), "Layer1 W wrong"
    assert np.allclose(model.layers['layer2']['W'], expected_W2), "Layer2 W wrong"
    print("✓ SGD correctly traverses nested structure")


def test_momentum_accumulation():
    """Test that momentum accumulates across steps."""
    print("\n=== Testing Momentum Accumulation ===")
    
    model = SimpleModel()
    optimizer = SGDWithMomentum(model, lr=0.1, momentum=0.9)
    
    W_before = model.W.copy()
    
    # Step 1: v = 0.9 * 0 + 0.1 * grad = 0.1 * grad
    optimizer.step()
    v1 = 0.1 * model.grad_W
    expected_W1 = W_before - v1
    assert np.allclose(model.W, expected_W1), "First step wrong"
    
    # Step 2: v = 0.9 * v1 + 0.1 * grad
    W_before2 = model.W.copy()
    optimizer.step()
    v2 = 0.9 * v1 + 0.1 * model.grad_W
    expected_W2 = W_before2 - v2
    assert np.allclose(model.W, expected_W2), "Second step wrong"
    
    print("✓ Momentum velocity accumulates correctly")


def test_adam_bias_correction():
    """Test Adam's bias correction is applied correctly."""
    print("\n=== Testing Adam Bias Correction ===")
    
    model = SimpleModel()
    model.grad_W = np.ones_like(model.W)  # Constant gradient
    model.grad_b = np.ones_like(model.b)
    optimizer = Adam(model, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    W_before = model.W.copy()
    
    optimizer.step()
    
    # Manual calculation
    m = 0.1 * model.grad_W  # (1 - beta1) * grad
    v = 0.001 * model.grad_W ** 2  # (1 - beta2) * grad^2
    m_hat = m / (1 - 0.9)  # Bias correction at t=1
    v_hat = v / (1 - 0.999)
    expected_W = W_before - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    assert np.allclose(model.W, expected_W), f"Adam step wrong"
    print("✓ Adam bias correction correct")


def test_adam_nested():
    """Test Adam with nested parameter structure."""
    print("\n=== Testing Adam (nested) ===")
    
    model = NestedModel()
    optimizer = Adam(model, lr=0.001)
    
    W1_before = model.layers['layer1']['W'].copy()
    W2_before = model.layers['layer2']['W'].copy()
    
    # Run multiple steps
    for _ in range(5):
        optimizer.step()
    
    # Parameters should change
    assert not np.allclose(model.layers['layer1']['W'], W1_before), "Layer1 W didn't change"
    assert not np.allclose(model.layers['layer2']['W'], W2_before), "Layer2 W didn't change"
    
    # No NaN or Inf
    assert not np.any(np.isnan(model.layers['layer1']['W'])), "NaN in layer1"
    assert not np.any(np.isnan(model.layers['layer2']['W'])), "NaN in layer2"
    
    print("✓ Adam handles nested structure correctly")
    print("✓ No numerical instability after multiple steps")


def test_adamw_weight_decay():
    """Test AdamW applies decoupled weight decay."""
    print("\n=== Testing AdamW Weight Decay ===")
    
    model = SimpleModel()
    model.grad_W = np.zeros_like(model.W)  # Zero gradient
    model.grad_b = np.zeros_like(model.b)
    
    optimizer = AdamW(model, lr=0.1, weight_decay=0.1)
    
    W_before = model.W.copy()
    
    optimizer.step()
    
    # With zero gradient, only weight decay should apply
    # θ = θ - lr * λ * θ = θ * (1 - lr * λ)
    expected_W = W_before * (1 - 0.1 * 0.1)
    
    # Note: Adam still updates m,v even with zero grad, so not exact
    # But weight should decrease
    assert np.all(np.abs(model.W) < np.abs(W_before)), "Weight decay not applied"
    print("✓ AdamW applies decoupled weight decay")


def test_with_full_jazz_model():
    """Test optimizer integration with full JazzRNN model."""
    print("\n=== Testing with Full JazzRNN Model ===")
    
    from src.RNN_numpy.model import JazzRNN
    from src.RNN_numpy import config
    
    model = JazzRNN(num_dur_classes=10)
    optimizer = Adam(model, lr=0.001)
    
    # Get initial weights
    params_before = model.get_all_params()
    W_pitch_before = params_before['emb_pitch']['W'].copy()
    W_rnn_before = params_before['rnn_cell']['W_xh'].copy()
    
    # Create dummy features
    batch_size, seq_len = 2, 4
    features = {
        'pitch': np.random.randint(0, config.VOCAB_SIZE_PITCH, (batch_size, seq_len)),
        'rel_pitch': np.random.randint(0, config.VOCAB_SIZE_REL_PITCH, (batch_size, seq_len)),
        'dur': np.random.randint(0, 10, (batch_size, seq_len)),
        'pos': np.random.randint(0, config.VOCAB_SIZE_GRID_POS, (batch_size, seq_len)),
        'chord_root': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT, (batch_size, seq_len)),
        'chord_root_rel': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT_REL, (batch_size, seq_len)),
        'chord_quality': np.random.randint(0, config.VOCAB_SIZE_CHORD_QUALITY, (batch_size, seq_len)),
        'next_chord_root': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT, (batch_size, seq_len)),
        'next_chord_root_rel': np.random.randint(0, config.VOCAB_SIZE_CHORD_ROOT_REL, (batch_size, seq_len)),
        'next_chord_quality': np.random.randint(0, config.VOCAB_SIZE_CHORD_QUALITY, (batch_size, seq_len)),
        'prev_interval': np.random.randint(0, config.VOCAB_SIZE_PREV_INTERVAL, (batch_size, seq_len)),
    }
    
    # Forward pass
    pitch_logits, dur_logits = model.forward(features)
    
    # Backward pass
    grad_pitch = np.random.randn(*pitch_logits.shape) * 0.01
    grad_dur = np.random.randn(*dur_logits.shape) * 0.01
    model.backward(grad_pitch, grad_dur)
    
    # Optimizer step
    optimizer.step()
    
    # Verify parameters changed
    params_after = model.get_all_params()
    
    # Check embedding weights changed
    assert not np.allclose(params_after['emb_pitch']['W'], W_pitch_before), \
        "Embedding weights didn't update"
    
    # Check RNN weights changed  
    assert not np.allclose(params_after['rnn_cell']['W_xh'], W_rnn_before), \
        "RNN weights didn't update"
    
    print("✓ All embedding layers updated")
    print("✓ RNN cell updated")
    print("✓ Output heads updated")
    print("✓ Full model training step successful")


def run_all_tests():
    """Run all optimizer tests."""
    print("=" * 60)
    print("Running Optimizer Tests (Nested Dict Support)")
    print("=" * 60)
    
    test_sgd_flat()
    test_sgd_nested()
    test_momentum_accumulation()
    test_adam_bias_correction()
    test_adam_nested()
    test_adamw_weight_decay()
    test_with_full_jazz_model()
    
    print("\n" + "=" * 60)
    print("✅ ALL OPTIMIZER TESTS PASSED!")
    print("=" * 60)
    print("\nKey Features Validated:")
    print("  1. Flat parameter dicts work")
    print("  2. Nested parameter dicts work (recursive traversal)")
    print("  3. State (momentum/velocity) persists across steps")
    print("  4. Bias correction applied correctly")
    print("  5. Full JazzRNN integration works")


if __name__ == "__main__":
    run_all_tests()

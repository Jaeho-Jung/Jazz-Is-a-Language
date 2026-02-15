"""
Optimizers for NumPy LSTM with nested dictionary support.

All optimizers handle both flat and nested parameter dictionaries.
"""

import numpy as np
from src.LSTM_numpy.config import LEARNING_RATE, MOMENTUM, RMSPROP_DECAY, ADAM_BETA1, ADAM_BETA2, EPSILON, WEIGHT_DECAY


class BaseOptimizer:
    """
    Base optimizer class with recursive parameter traversal.
    
    Handles nested parameter dictionaries of arbitrary depth:
        {'layer': {'W': array, 'b': array}}
    """
    
    def __init__(self, model, lr=LEARNING_RATE):
        self.model = model
        self.lr = lr
    
    def step(self):
        """Update all parameters. Subclasses must implement."""
        raise NotImplementedError
    
    def _is_leaf(self, value):
        """Check if value is a leaf parameter (numpy array)."""
        return isinstance(value, np.ndarray)
    
    def _traverse_and_update(self, params, grads, state_dicts, update_fn, path=""):
        """
        Recursively traverse nested dicts and apply update function to leaves.
        
        Args:
            params: Parameter dict (nested or flat)
            grads: Gradient dict (same structure as params)
            state_dicts: List of state dictionaries (e.g., [m, v] for Adam)
            update_fn: Function(param, grad, *states, path) -> updated_param, *updated_states
            path: Current path in the nested structure (for state key generation)
        """
        for key in params:
            full_key = f"{path}.{key}" if path else key
            
            if self._is_leaf(params[key]):
                # Initialize states if needed
                for state in state_dicts:
                    if full_key not in state:
                        state[full_key] = np.zeros_like(params[key])
                
                # Get current states
                current_states = [state[full_key] for state in state_dicts]
                
                # Apply update
                result = update_fn(params[key], grads[key], *current_states, full_key)
                
                # Unpack result: (updated_param, *updated_states)
                params[key] = result[0]
                for i, state in enumerate(state_dicts):
                    state[full_key] = result[i + 1]
            else:
                # Recurse into nested dict
                self._traverse_and_update(params[key], grads[key], state_dicts, update_fn, full_key)


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent.
    
    Update rule: θ = θ - lr * ∇θ
    """
    
    def __init__(self, model, lr=LEARNING_RATE):
        super().__init__(model, lr)
    
    def step(self):
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        
        def update(param, grad, path):
            return (param - self.lr * grad,)
        
        self._traverse_and_update(params, grads, [], update)
        self.model.set_all_params(params)


class SGDWithMomentum(BaseOptimizer):
    """
    SGD with Momentum.
    
    Update rule:
        v = β * v + lr * ∇θ
        θ = θ - v
    """
    
    def __init__(self, model, lr=LEARNING_RATE, momentum=MOMENTUM):
        super().__init__(model, lr)
        self.momentum = momentum
        self.v = {}
    
    def step(self):
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        
        def update(param, grad, v, path):
            v_new = self.momentum * v + self.lr * grad
            param_new = param - v_new
            return (param_new, v_new)
        
        self._traverse_and_update(params, grads, [self.v], update)
        self.model.set_all_params(params)


class AdaGrad(BaseOptimizer):
    """
    AdaGrad - Adaptive Gradient Algorithm.
    
    Update rule:
        G = G + ∇θ²
        θ = θ - lr / √(G + ε) * ∇θ
    
    Note: Learning rate monotonically decreases (never recovers).
    """
    
    def __init__(self, model, lr=LEARNING_RATE, epsilon=EPSILON):
        super().__init__(model, lr)
        self.epsilon = epsilon
        self.G = {}
    
    def step(self):
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        
        def update(param, grad, G, path):
            G_new = G + grad ** 2
            param_new = param - self.lr / np.sqrt(G_new + self.epsilon) * grad
            return (param_new, G_new)
        
        self._traverse_and_update(params, grads, [self.G], update)
        self.model.set_all_params(params)


class RMSProp(BaseOptimizer):
    """
    RMSProp - Root Mean Square Propagation.
    
    Update rule:
        E = ρ * E + (1 - ρ) * ∇θ²
        θ = θ - lr / √(E + ε) * ∇θ
    
    Fixes AdaGrad's monotonically decreasing learning rate.
    """
    
    def __init__(self, model, lr=LEARNING_RATE, epsilon=EPSILON, decay=RMSPROP_DECAY):
        super().__init__(model, lr)
        self.epsilon = epsilon
        self.decay = decay
        self.E = {}
    
    def step(self):
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        
        def update(param, grad, E, path):
            E_new = self.decay * E + (1 - self.decay) * grad ** 2
            param_new = param - self.lr / np.sqrt(E_new + self.epsilon) * grad
            return (param_new, E_new)
        
        self._traverse_and_update(params, grads, [self.E], update)
        self.model.set_all_params(params)


class Adam(BaseOptimizer):
    """
    Adam - Adaptive Moment Estimation.
    
    Combines momentum (first moment) and RMSProp (second moment) with bias correction.
    
    Update rule:
        m = β1 * m + (1 - β1) * ∇θ
        v = β2 * v + (1 - β2) * ∇θ²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        θ = θ - lr * m_hat / (√v_hat + ε)
    """
    
    def __init__(self, model, lr=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2, epsilon=EPSILON):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self):
        self.t += 1
        
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        
        # Bias correction denominators (computed once per step)
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        
        def update(param, grad, m, v, path):
            # Update biased first and second moment estimates
            m_new = self.beta1 * m + (1 - self.beta1) * grad
            v_new = self.beta2 * v + (1 - self.beta2) * grad ** 2
            
            # Bias-corrected estimates
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            
            # Update parameter
            param_new = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            return (param_new, m_new, v_new)
        
        self._traverse_and_update(params, grads, [self.m, self.v], update)
        self.model.set_all_params(params)


class AdamW(BaseOptimizer):
    """
    AdamW - Adam with decoupled weight decay.
    
    Weight decay is applied directly to parameters, not through gradients.
    This is the recommended optimizer for training transformers and modern networks.
    
    Update rule:
        m = β1 * m + (1 - β1) * ∇θ
        v = β2 * v + (1 - β2) * ∇θ²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        θ = θ - lr * (m_hat / (√v_hat + ε) + λ * θ)
    """
    
    def __init__(self, model, lr=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2, 
                 epsilon=EPSILON, weight_decay=WEIGHT_DECAY):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self):
        self.t += 1
        
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        
        def update(param, grad, m, v, path):
            m_new = self.beta1 * m + (1 - self.beta1) * grad
            v_new = self.beta2 * v + (1 - self.beta2) * grad ** 2
            
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            
            # Decoupled weight decay: apply directly to param
            param_new = param - self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param)
            
            return (param_new, m_new, v_new)
        
        self._traverse_and_update(params, grads, [self.m, self.v], update)
        self.model.set_all_params(params)

import numpy as np
from src.Transformer_numpy import config
from typing import Callable

class BaseOptimizer:
    """
    Base optimizer class
    """
    def __init__(self, model, lr=config.LEARNING_RATE):
        self.model = model
        self.lr = lr

    def _update(self, param: np.ndarray, grad: np.ndarray, *args: np.ndarray, path: str) -> tuple[np.ndarray]:
        """Update parameters"""
        raise NotImplementedError

    def step(self, update_fn: Callable, state_dicts: list) -> None:
        params = self.model.get_all_params()
        grads = self.model.get_all_grads()
        self._traverse_and_update(params, grads, state_dicts, update_fn)
        self.model.set_all_params(params)

    def _is_leaf(self, val: np.ndarray) -> bool:
        return isinstance(val, np.ndarray)

    def _traverse_and_update(self, params: dict, grads: dict, state_dicts: list, update_fn: Callable, path: str = "") -> None:
        """
        Recursively traverse nested dicts and apply update functions to leaves.

        Args:
            params: Parameter dict (nested or flat)
            grads: Gradient dict (same structure as params)
            state_dicts: List of state dictionaries (e.g., [m, v] for Adam)
            update_fn: Function(param, grad, *states, path) -> updated_param, *updated_states
            path: Current path in the nested structure (for state key generation)
        """
        for key, val in params.items():
            current_path = f"{path}.{key}" if path else key
            
            # Recursive case
            if not self._is_leaf(val):
                self._traverse_and_update(val, grads[key], state_dicts, update_fn, current_path)
                continue
            
            # Leaf case (Parameter)
            grad = grads[key]
            
            # Gather states for this param
            current_states = []
            for state in state_dicts:
                if key not in state:
                    state[key] = np.zeros_like(val)
                current_states.append(state[key])
            
            # Apply update
            results = update_fn(val, grad, *current_states, path=current_path)
            
            # Update param
            params[key] = results[0]
            
            # Update states
            for i, state in enumerate(state_dicts):
                state[key] = results[i + 1]

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent.

    Update rule: θ = θ - lr * ∇θ
    """
    def __init__(self, model, lr=config.LEARNING_RATE):
        super().__init__(model, lr)

    def _update(self, param: np.ndarray, grad: np.ndarray, path: str) -> tuple[np.ndarray]:
        return (param - self.lr * grad,)

    def step(self) -> None:
        super().step(self._update, [])

class SGDWithMomentum(BaseOptimizer):
    """
    SGD with Momentum.

    Update rule:
        v = β * v + lr * ∇θ
        θ = θ - v
    """
    def __init__(self, model, lr=config.LEARNING_RATE, momentum=config.MOMENTUM):
        super().__init__(model, lr)
        self.momentum = momentum
        self.v = {}

    def _update(self, param: np.ndarray, grad: np.ndarray, v: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
        v_new = self.momentum * v + self.lr * grad
        param_new = param - v_new
        return (param_new, v_new)
    
    def step(self) -> None:
        super().step(self._update, [self.v])

class AdaGrad(BaseOptimizer):
    """
    AdaGrad - Adaptive Gradient Algorithm.

    Update rule:
        G = G + ∇θ²
        θ = θ - lr / √(G + ε) * ∇θ
    """
    def __init__(self, model, lr=config.LEARNING_RATE, epsilon=config.EPSILON):
        super().__init__(model, lr)
        self.epsilon = epsilon
        self.G = {}

    def _update(self, param: np.ndarray, grad: np.ndarray, G: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
        G_new = G + grad ** 2
        param_new = param - self.lr / np.sqrt(G_new + self.epsilon) * grad
        return (param_new, G_new)

    def step(self) -> None:
        super().step(self._update, [self.G])

class RMSProp(BaseOptimizer):
    """
    RMSProp - Root Mean Square Propagation.

    Update rule:
        E[g²]_t = β * E[g²]_(t-1) + (1 - β) * g²_t
        θ = θ - lr / √(E[g²]_t + ε) * g_t
    """
    def __init__(self, model, lr=config.LEARNING_RATE, epsilon=config.EPSILON, decay=config.RMSPROP_DECAY):
        super().__init__(model, lr)
        self.epsilon = epsilon
        self.decay = decay
        self.E = {}

    def _update(self, param: np.ndarray, grad: np.ndarray, E: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
        E_new = self.decay * E + (1 - self.decay) * grad ** 2
        param_new = param - self.lr / np.sqrt(E_new + self.epsilon) * grad
        return (param_new, E_new)

    def step(self) -> None:
        super().step(self._update, [self.E])

class Adam(BaseOptimizer):
    """
    Adam - Adaptive Moment Estimation.
    
    Update rule:
        m = β1 * m + (1 - β1) * ∇θ
        v = β2 * v + (1 - β2) * ∇θ²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        θ = θ - lr * m_hat / (√v_hat + ε)
    """
    def __init__(self, model, lr=config.LEARNING_RATE, beta1=config.ADAM_BETA1, beta2=config.ADAM_BETA2, epsilon=config.EPSILON):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def _update(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.t += 1
        m_new = self.beta1 * m + (1 - self.beta1) * grad
        v_new = self.beta2 * v + (1 - self.beta2) * grad ** 2
        m_hat = m_new / (1 - self.beta1 ** self.t)
        v_hat = v_new / (1 - self.beta2 ** self.t)
        param_new = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return (param_new, m_new, v_new)

    def step(self) -> None:
        super().step(self._update, [self.m, self.v])

class AdamW(BaseOptimizer):
    """
    AdamW - Adam with decoupled weight decay.
    
    Update rule:
        m = β1 * m + (1 - β1) * ∇θ
        v = β2 * v + (1 - β2) * ∇θ²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        θ = θ - lr * (m_hat / (√v_hat + ε) + λ * θ)
    """
    def __init__(self, model, lr=config.LEARNING_RATE, beta1=config.ADAM_BETA1, beta2=config.ADAM_BETA2, epsilon=config.EPSILON, weight_decay=config.WEIGHT_DECAY):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def _update(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.t += 1
        m_new = self.beta1 * m + (1 - self.beta1) * grad
        v_new = self.beta2 * v + (1 - self.beta2) * grad ** 2
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param_new = param - self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param)
        return (param_new, m_new, v_new)

    def step(self) -> None:
        super().step(self._update, [self.m, self.v])
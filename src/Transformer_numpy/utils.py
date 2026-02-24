import numpy as np

# Sigmoid, derivative, tanh, softmax, cross entropy, to numpy, to onehot

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def gelu(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.44715 * x ** 3)))

def gelu_derivative(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.44715 * x ** 3)
    tanh_val = np.tanh(inner)
    d_inner = c * (1 + 1.34145 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * d_inner * (1 - tanh_val ** 2)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-9) -> float:
    return -np.mean(y_true * np.log(y_pred + epsilon))

def cross_entropy_loss_grad(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    return (y_pred - y_true) / y_pred.shape[0]

def to_numpy(x: np.ndarray) -> np.ndarray:
    if hasattr(x, 'numpy'):
        return x.numpy()
    return x

def to_one_hot(x: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((x.shape[0], num_classes))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
    
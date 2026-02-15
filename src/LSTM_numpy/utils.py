import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.
    """
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    """
    Tanh activation function.
    """
    return np.tanh(z)

def tanh_derivative(z):
    """
    Derivative of tanh function.
    """
    return 1 - np.tanh(z)**2

def softmax(z):
    """
    Numerically stable softmax function.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
def cross_entropy_loss(logits, targets, epsilon=1e-12):
    """
    Cross-entropy loss function.
    """
    m = targets.shape[0]
    probs = softmax(logits)
    loss = (-1/m) * np.sum(targets * np.log(probs + epsilon))
    return loss

def cross_entropy_grad(logits, target, epsilon=1e-12):
    """
    Gradient of cross-entropy loss.
    """
    m = target.shape[0]
    probs = softmax(logits)
    grad = (1/m) * (probs - target)
    return grad
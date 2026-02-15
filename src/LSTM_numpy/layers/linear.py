import numpy as np


class Linear:
    """
    Fully connected (dense) layer.
    
    Computes: y = x @ W.T + b
    """
    
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        
        # Xavier/He initialization
        self.W = np.random.randn(output_features, input_features) * np.sqrt(2.0 / input_features)
        self.b = np.zeros(output_features)
        
        # Gradient storage
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input, shape (batch, input_features)
        
        Returns:
            z: Output, shape (batch, output_features)
        """
        self.x = x
        z = x @ self.W.T + self.b
        return z

    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient from next layer, shape (batch, output_features)
        
        Returns:
            grad_x: Gradient w.r.t. input, shape (batch, input_features)
        """
        # Gradients
        grad_x = grad_output @ self.W
        self.grad_W = grad_output.T @ self.x
        self.grad_b = np.sum(grad_output, axis=0)
        
        return grad_x
    
    def get_params(self):
        """Return parameters for optimizer."""
        return {'W': self.W, 'b': self.b}
    
    def get_grads(self):
        """Return gradients for optimizer."""
        return {'W': self.grad_W, 'b': self.grad_b}
    
    def set_params(self, params):
        """Update parameters from optimizer."""
        self.W = params['W']
        self.b = params['b']

import numpy as np

__all__ = ["sign_activation",
            "linear_regression_activation",
            "sigmoid_activation"]
            
def sign_activation(z):
    return np.sign(z)

def linear_regression_activation(z):
    return z

def sigmoid_activation(z):
    return 1.0/(1.0 + np.exp(-z))

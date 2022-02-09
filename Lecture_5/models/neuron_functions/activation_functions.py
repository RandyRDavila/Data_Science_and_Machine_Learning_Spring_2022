
import numpy as np


def sign_activation(z):
    return np.sign(z)

def linear_regression_activation(z):
    return z

def sigmoid_activation(z):
    return 1.0/(1.0 + np.exp(-z))

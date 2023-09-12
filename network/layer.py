"""A module containing the Layer class

"""

from .activation_funcs import activation_functions
import numpy as np

__all__ = ["Layer"]


class Layer:
    """
    A class to store all the information needed for a network layer

    Attributes
    ----------
    W: ndarray
        The weight matrix connecting this layer and the previous layer
        Shape (m_{l} x m_{l-1})
    b: ndarray
        A column vector containing biases for this layer
        Shape (m_{l} x 1)
    f: callable
        The activation function for this layer
    f_grad: callable
        The gradient of the activation function
    m: int
        The number of neurons in the layer
    A: ndarray
        A column vector containing the activations
        of the most recent feed-forward
        Shape (m_{l} x 1)
    Z: ndarray
        A column vector containing the pre-activations
        of the most recent feed-forward

    Notes
    -----
    Activation: A
    Pre-Activation: Z
    Weight Matrix: W
    Bias Vector: b
    Activation-function: f()

    A_{l} = f(Z_{l})
    Z_{l} = W_{l} x A_{l-1} + b_{l}
    """
    def __init__(self, m: int, activation: str = "relu") -> None:
        self.W = np.array([])
        self.b = np.array([])
        self.f, self.f_grad\
            = activation_functions[activation]
        self.m = m
        self.A = np.array([])
        self.Z = np.array([])
        self.d_A = np.array([])
        self.d_Z = np.array([])
        self.d_W = np.array([])
        self.d_b = np.array([])

    def reset(self):
        self.d_W = 0
        self.d_b = 0

"""A module containing the Layer class

"""

from .activation_funcs import activation_functions

__all__ = ["Layer"]


class Layer:
    """
    A class to store all the information needed for a network layer

    Attributes
    ----------
    weights: ndarray
        The weight matrix connecting this layer and the previous layer
        Shape (m_{l} x m_{l-1})
    biases: ndarray
        A column vector containing biases for this layer
        Shape (m_{l} x 1)
    activation_func: callable
        The activation function for this layer
    activation_grad: callable
        The gradient of the activation function
    m: int
        The number of neurons in the layer
    activation: ndarray
        A column vector containing the activations
        of the most recent feed-forward
        Shape (m_{l} x 1)
    pre_activation: ndarray
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
        self.weights = None
        self.biases = None
        self.activation_func, self.activation_grad\
            = activation_functions[activation]
        self.m = m
        self.activation = None
        self.pre_activation = None

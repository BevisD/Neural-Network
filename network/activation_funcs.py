"""Module containing activation functions and their gradients
for neural-network layers

"""

import numpy as np

__all__ = ["activation_functions"]


def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies the ReLU function to the input

    Output is 0 for negative inputs and unchanged for positive inputs

    Parameters
    ----------
    x : ndarray
        The input for the ReLU function

    Returns
    -------
    ndarray
        The result of applying the ReLU function to the input

    """
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the relu function at x

    Output is 0 for negative inputs and 1 for positive inputs

    Parameters
    ----------
    x: ndarray
        Input vector for relu gradient

    Returns
    -------
    grad: ndarray
        Gradient of relu at x
    """
    return (x > 0).astype(int)


def linear(x: np.ndarray) -> np.ndarray:
    """
    The linear activation function returns the input

    Output is unchanged

    Parameters
    ----------
    x: ndarray
        The input

    Returns
    -------
    x: ndarray
        The output

    """
    return x


def linear_grad(x: np.ndarray) -> np.ndarray:
    """Computes the gradient of the linear function

    The gradient of the linear function is always 1

    Parameters
    ----------
    x: ndarray
        The value to compute the gradient at

    Returns
    -------
    ndarray
        always 1
    """
    return np.ones_like(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function to the input

    Maps any real number to the range (-1, 1)

    -inf -> 0
    0 -> 0.5
    inf -> 1

    Parameters
    ----------
    x: ndarray
        The pre-activation matrix

    Returns
    -------
    ndarray
        The activation matrix

    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient of the sigmoid function at x

    A quirk of the sigmoid function is that the gradient of s is
    s * (1 - s)

    Parameters
    ----------
    x: ndarray
        The pre-activation matrix

    Returns
    -------
    ndarray
        The activation matrix

    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1 / np.cosh(x) ** 2


activation_functions = {
    "relu": (relu, relu_grad),
    "linear": (linear, linear_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (np.tanh, tanh_grad)
}

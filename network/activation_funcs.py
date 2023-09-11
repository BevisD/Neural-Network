"""Module containing activation functions for neural-network layers

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


activation_functions = {
    "relu": (relu, relu_grad),
    "linear": (linear, linear_grad)
}

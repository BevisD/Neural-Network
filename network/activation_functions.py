"""Module containing activation functions for neural-network layers

"""

import numpy as np

__all__ = ["relu", "relu_grad"]


def relu(x: np.ndarray):
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


def relu_grad(x: np.ndarray):
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

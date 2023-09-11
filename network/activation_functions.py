"""Module containing activation functions for neural-network layers

"""

import numpy as np

__all__ = ["relu"]


def relu(x):
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

"""Module containing cost/loss functions and their gradients

"""

import numpy as np

__all__ = ["cost_functions", "mean_square_error"]


def mean_square_error(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the MSE of output and target

    Parameters
    ----------
    t: ndarray
        Target vector
    y: ndarray
        Output vector

    Returns
    -------
    cost: float
        Total cost of the output and target

    """
    assert t.shape == y.shape
    diff = y - t
    cost = np.sum(diff**2)
    return cost


def mean_square_grad(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the MSE for output and target

    Parameters
    ----------
    t: ndarray
        Target vector
    y: ndarray
        Output vector

    Returns
    -------
    grad: ndarray
        Gradient of cost function
    """
    assert t.shape == y.shape
    diff = y - t
    grad = 2 * diff
    return grad


cost_functions = {"MSE": (mean_square_error, mean_square_grad)}

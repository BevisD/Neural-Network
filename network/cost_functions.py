import numpy as np

__all__ = ["mean_square_error", "mean_square_grad"]


def mean_square_error(t, y):
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
    loss: float
        Total cost of the output and target

    """
    assert t.shape == y.shape
    diff = y - t
    cost = np.sum(diff**2) / t.shape[0]
    return cost


def mean_square_grad(t, y):
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
    grad = 2 * diff / t.shape[0]

    if np.ndim(grad) == 3:
        grad = np.sum(grad, axis=0)
    return grad

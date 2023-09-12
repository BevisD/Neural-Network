"""A module for the NeuralNetwork class

"""

import numpy as np
from .activation_funcs import *
from .cost_funcs import *
from .layer import *

__all__ = ["NeuralNetwork", "create_mini_batches"]


def create_mini_batches(data_size: int, batch_size: int):
    """
    Creates batches of indices

    Parameters
    ----------
    data_size: int
        The total number of training data points
    batch_size
        The size of the batch to split into

    Returns
    -------
    mini_batches: list

    Examples
    --------
    >>> create_mini_batches(10, 4)
    [array([6, 2, 9, 7]), array([0, 4, 3, 8]), array([1, 5])] # random

    """
    mini_batches = []

    # Shuffle the data
    perm = np.random.permutation(data_size)

    num_batches = data_size // batch_size
    for i in range(num_batches):
        mini_batch = perm[i * batch_size: (i + 1) * batch_size]

        mini_batches.append(mini_batch)

    # Handle last batch if it's smaller than batch size
    if data_size % batch_size != 0:
        mini_batch = perm[num_batches * batch_size:]
        mini_batches.append(mini_batch)

    return mini_batches


class NeuralNetwork:
    """
    A Neural Network class

    Attributes
    ----------
    shape: list[int]
        The number of neurons in each layer
    layers: list[Layer]
        List containing the layers of the network
    N: int
        Number of layers
    losses: list
        History of total loss at each epoch

    Methods
    -------
    add_layer(layer: Layer) -> None
        Appends a layer to the network
    feed_forward(X: ndarray) -> ndarray
        Calculates the networks output from the input
    fit(X: np.ndarray, Y: np.ndarray,
        eta: float = 0.01, cost: str = "MSE",
        epochs: int = 100, verbose: bool = True) -> None:
        Trains the network on inputs and outputs using SGD

    """

    def __init__(self, m: int) -> None:
        self.shape = [m]
        self.layers = [Layer(m)]
        self.N = 1
        self.losses = []

    def add_layer(self, layer: Layer) -> None:
        """
        Appends a layer object to the network

        Parameters
        ----------
        layer: Layer
            The layer to append to the network

        """
        layer.W = np.random.normal(size=(layer.m, self.shape[-1]))
        layer.b = np.random.normal(size=(layer.m, 1))

        self.shape.append(layer.m)
        self.layers.append(layer)
        self.N += 1

    def reset_layers(self):
        for layer in self.layers:
            layer.reset()

    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the network for the input given

        Parameters
        ----------
        X: ndarray
            The input to pass through the network
            Shape (N, m_0, 1) or (m_0, 1)

        Returns
        -------
        activation: ndarray
            The activation of the output layer of the network
        """
        A = X.copy()
        self.layers[0].A = A
        for layer in self.layers[1:]:
            Z = np.matmul(layer.W, A) + layer.b
            A = layer.f(Z)

            layer.Z = Z
            layer.A = A
        return A

    def fit(self, X: np.ndarray, Y: np.ndarray,
            eta: float = 0.01, cost: str = "MSE",
            epochs: int = 100, batch_size: int = 1,
            verbose: bool = True) -> None:
        """

        Parameters
        ----------
        X: ndarray
            Set of input training data
        Y: ndarray
            Set of target training data
        eta: float
            Learning rate
        cost: str
            Name of the cost function to use
        epochs: int
            Number of iterations of training data
        batch_size: int
            Number of data points to train on before
            updating weights and biases
        verbose: bool
            Option to print out training statistics

        """

        self.losses = []
        for epoch in range(epochs):
            loss = 0
            mini_batches = create_mini_batches(X.shape[0], batch_size)
            for mini_batch in mini_batches:
                X_mini = X[mini_batch]
                Y_mini = Y[mini_batch]
                loss += self.fit_batch(X_mini, Y_mini, cost=cost, eta=eta)

            self.losses.append(loss)
            if verbose:
                print(f"Epoch {epoch},\tLoss: {loss:.4e}")

    def fit_batch(self, X, Y, cost: str = "MSE", eta: float = 0.01) -> float:
        L, L_grad = cost_functions[cost]
        self.reset_layers()

        loss = 0
        for x, y in zip(X, Y):
            # FORWARD STEP
            self.feed_forward(x)

            # BACKWARD STEP
            # Initialise backward step by calculating activation differential
            self.layers[-1].d_A = L_grad(y, self.layers[-1].A)
            loss += L(y, self.layers[-1].A)

            for L in range(self.N - 1, 0, -1):
                this_layer = self.layers[L]
                prev_layer = self.layers[L - 1]

                # Calculate weights and biases gradients
                # Update activation differential
                this_layer.d_Z = this_layer.d_A * this_layer.f_grad(this_layer.Z)
                this_layer.d_W += np.matmul(this_layer.d_Z, prev_layer.A.T)
                this_layer.d_b += this_layer.d_Z
                prev_layer.d_A = np.matmul(this_layer.W.T, this_layer.d_Z)

        # Update weights and biases
        for layer in self.layers:
            layer.W -= eta * layer.d_W
            layer.b -= eta * layer.d_b

        return loss

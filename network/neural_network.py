"""A module for the NeuralNetwork class

"""

import numpy as np
from .activation_funcs import *
from .cost_funcs import *
from .layer import *

__all__ = ["NeuralNetwork"]


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
        layer.weights = np.random.normal(size=(layer.m, self.shape[-1]))
        layer.biases = np.random.normal(size=(layer.m, 1))
        self.shape.append(layer.m)
        self.layers.append(layer)
        self.N += 1

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
        activation = X.copy()
        self.layers[0].activation = activation
        for layer in self.layers[1:]:
            pre_activation = np.matmul(layer.weights, activation) + layer.biases
            activation = layer.activation_func(pre_activation)

            layer.pre_activation = pre_activation
            layer.activation = activation
        return activation

    def fit(self, X: np.ndarray, Y: np.ndarray,
            eta: float = 0.01, cost: str = "MSE",
            epochs: int = 100, verbose: bool = True) -> None:
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
        verbose: bool
            Option to print out training statistics

        """
        cost_func, cost_grad = cost_functions[cost]

        self.losses = []
        for epoch in range(epochs):
            loss = 0
            for x, y in zip(X, Y):
                # FORWARD STEP
                self.feed_forward(x)

                # BACKWARD STEP
                # Initialise backward step by calculating activation differential
                d_activation = cost_grad(y, self.layers[-1].activation)
                loss += cost_func(y, self.layers[-1].activation)

                for L in range(self.N - 1, 0, -1):
                    layer = self.layers[L]
                    prev_layer = self.layers[L - 1]

                    # Calculate weights and biases gradients
                    # Update activation differential
                    d_pre_activation = d_activation * layer.activation_grad(layer.pre_activation)
                    d_weights = np.matmul(d_pre_activation, prev_layer.activation.T)
                    d_biases = d_pre_activation
                    d_activation = np.matmul(layer.weights.T, d_pre_activation)

                    # Update weights and biases
                    layer.weights -= eta * d_weights
                    layer.biases -= eta * d_biases

            self.losses.append(loss)
            if verbose:
                print(f"Epoch {epoch},\tLoss: {loss:.4e}")

"""A module for the NeuralNetwork class
"""

import numpy as np
from .activation_functions import *
from .cost_functions import *

__all__ = ["NeuralNetwork"]


class NeuralNetwork:
    """
    A Neural Network

    Attributes
    ----------
    N : int
        Number of layers in the network
    weights : list[ndarray]
        A list of the weight matrices
    biases :
        A list of the bias matrices

    Methods
    -------
    feed_forward(inputs)
        Produces the network's output from the input
    """
    def __init__(self, shape: list[int]):
        """
        Parameters
        ----------
        shape : list[int]
            A list of the number of neurons in each layer
        """
        self.N = len(shape)
        self.weights = [np.random.uniform(0, 1, size=(shape[i+1], shape[i]))
                        for i in range(self.N-1)]
        self.biases = [np.random.uniform(0, 1, size=(shape[i], 1))
                       for i in range(1, self.N)]
        self.cost = mean_square_error
        self.cost_grad = mean_square_grad

    def feed_forward(self, inputs: np.ndarray):
        """
        Parameters
        ----------
        inputs : ndarray
            The input for each neuron in the input layer

        Returns
        -------
        outputs : ndarray
            The output for each neuron in the output layer
        """
        outputs = inputs.copy()
        for i in range(self.N-1):
            outputs = relu(self.weights[i] @ outputs + self.biases[i])

        return outputs

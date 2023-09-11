"""A module for the NeuralNetwork class
"""

import numpy as np
from .activation_funcs import *
from .cost_funcs import *
from .layer import *

__all__ = ["NeuralNetwork"]


class NeuralNetwork:
    def __init__(self, m: int) -> None:
        self.shape = [m]
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        layer.weights = np.random.normal(size=(layer.m, self.shape[-1]))
        layer.biases = np.random.normal(size=(layer.m, 1))
        self.shape.append(layer.m)
        self.layers.append(layer)

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        activation = x.copy()
        for layer in self.layers:
            activation = layer.activation_func(
                np.matmul(layer.weights, activation) + layer.biases
            )
        return activation

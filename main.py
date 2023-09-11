import numpy as np
import matplotlib.pyplot as plt
from network import *


def main():
    np.random.seed(1)

    NN = NeuralNetwork(2)
    NN.add_layer(Layer(4))
    NN.add_layer(Layer(1, activation="linear"))

    X = np.array([[[1], [1]],
                  [[1], [0]],
                  [[0], [1]],
                  [[0], [0]]])
    Y = np.array([[[0]],
                  [[1]],
                  [[1]],
                  [[0]]])

    NN.fit(X, Y, epochs=100)


if __name__ == "__main__":
    main()

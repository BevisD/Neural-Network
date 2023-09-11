import numpy as np
import matplotlib.pyplot as plt
from network import *


def main():
    np.random.seed(1)

    NN = NeuralNetwork(2)
    NN.add_layer(Layer(4))
    NN.add_layer(Layer(1, activation="linear"))

    x = np.array([[[1], [1]],
                  [[1], [0]],
                  [[0], [1]],
                  [[0], [0]]])
    t = np.array([[[0], [1]],
                  [[1], [0]],
                  [[1], [1]],
                  [[0], [1]]])

    print(NN.feed_forward(x))


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from network import *


def main():
    np.random.seed(1)

    NN = NeuralNetwork(2)
    NN.add_layer(Layer(2))
    NN.add_layer(Layer(1, activation="linear"))

    X_train = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    Y_train = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    NN.fit(X_train, Y_train, 500, verbose=False)
    print(NN.feed_forward(X_train))


if __name__ == "__main__":
    main()

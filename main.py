import numpy as np
import matplotlib.pyplot as plt
from network import *


def main():
    np.random.seed(1)

    NN_r = NeuralNetwork(2)
    NN_r.add_layer(Layer(3, activation="relu"))
    NN_r.add_layer(Layer(1, activation="linear"))

    NN_t = NeuralNetwork(2)
    NN_t.add_layer(Layer(3, activation="tanh"))
    NN_t.add_layer(Layer(1, activation="linear"))

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

    NN_r.fit(X_train, Y_train, 1000, eta=0.1, verbose=False)
    NN_t.fit(X_train, Y_train, 1000, eta=0.1, verbose=False)

    plt.plot(NN_r.losses, label="ReLU")
    plt.plot(NN_t.losses, label="tanh")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

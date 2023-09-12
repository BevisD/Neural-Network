import numpy as np
import matplotlib.pyplot as plt
from network import *


def main():
    np.random.seed(1)

    NN_1 = NeuralNetwork(2)
    NN_1.add_layer(Layer(3))
    NN_1.add_layer(Layer(1, activation="linear"))
    NN_2 = NN_1.copy()
    NN_4 = NN_1.copy()

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

    NN_1.fit(X_train, Y_train, 100, verbose=False)
    NN_2.fit(X_train, Y_train, 100, batch_size=2, verbose=False)
    NN_4.fit(X_train, Y_train, 100, batch_size=4, verbose=False)
    plt.plot(NN_1.losses, label="Batch-size-1")
    plt.plot(NN_2.losses, label="Batch-size-2")
    plt.plot(NN_4.losses, label="Batch-size-4")

    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

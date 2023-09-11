from network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from network import relu
from network.cost_functions import *


def main():
    np.random.seed(0)
    NN = NeuralNetwork([2, 4, 1])
    t = np.array([[1, 2],
                  [2, 1],
                  [3, 4]])
    y = np.array([[3, 3],
                  [2, 3],
                  [1, 2]])
    print(mean_square_grad(t, y))


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from network import *


def main():
    np.random.seed(0)
    NN = NeuralNetwork([2, 4, 2])
    x = np.array([[[1], [1]],
                  [[1], [0]],
                  [[0], [1]],
                  [[0], [0]]])
    t = np.array([[[0], [1]],
                  [[1], [0]],
                  [[1], [1]],
                  [[0], [1]]])


if __name__ == "__main__":
    main()

from .activation_funcs import activation_functions

__all__ = ["Layer"]


class Layer:
    def __init__(self, m, activation: str = "relu") -> None:
        self.weights = None
        self.biases = None
        self.activation_func, self.activation_grad\
            = activation_functions[activation]
        self.m = m

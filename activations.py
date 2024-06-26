"""Activations should work over matrices ? or only floats?"""

from math import exp


def sigmoid(x:float)->float:
    return 1/(1+exp(-x))


def sigmoid_der(x:float)->float:
    return sigmoid(x) * (1 - sigmoid(x))
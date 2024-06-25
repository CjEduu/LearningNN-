"""Activations should work over matrices ? or only floats?"""

from math import exp


def sigmoid(x:float)->float:
    return 1/(1+exp(-x))
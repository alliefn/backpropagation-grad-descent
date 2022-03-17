import imp
from activation_functions import *

def diffLinearActvFunc(x):
    # return 1 if >= 0 else 0
    if x >= 0:
        return 1
    else:
        return 0

def diffSigmoidActvFunc(x):
    return sigmoid(x) * (1 - sigmoid(x))

def diffSoftmaxActvFunc(p, j, c):
    # p is probability
    # j is the class
    # c is target class
    if (j != c):
        return p
    else:
        return -(1-p)
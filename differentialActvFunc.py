import imp
from activation_functions import *

def diffLinearActvFunc(x):
    # return 1 if >= 0 else 0
    return 1

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

def diffReluActvFunc(x):
    if x >= 0:
        return 1
    else:
        return 0

# lossfunction for linear, sigmoid, and relu: E = (1/2) * sum(y - y_hat)^2
def lossFuncSumofSquaredErrors(y, y_hat):
    return (1/2) * sum((y - y_hat) ** 2)

# lossfunction for softmax: E = -log(p_c)
def lossFuncCrossEntropy(p, j, c):
    return -log(p[j])

# derivative of lossfunction for softmax
def diffLossFuncCrossEntropy(p, j, c):
    return -(1/(p[j]*ln(10)))
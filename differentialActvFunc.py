from activation_functions import *
import numpy as np

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

def difflossFuncSumofSquaredErrors(y, y_hat):
    return y - y_hat

# lossfunction for softmax: E = -log(p_c)
# def lossFuncCrossEntropy(p, j):
#     # [DESC]
#     # Lossfunction for softmax
#     # p is the probability
#     # j is the neuron index
#     return -log(p[j])

# derivative of lossfunction for softmax
def diffLossFuncCrossEntropy(p, j):
    # [DESC]
    # Derivative of lossfunction for softmax
    # p is the probability
    # j is the neuron index
    return -(1/(p[j] * np.log(10)))


def outputActivationFunc(t,o,x,j,i):
    # [DESC]
    # Neuron output activation function other than softmax
    # t is the target
    # o is the output
    # x is the input
    # j is the neuron index
    # i is the input index
    return -(t[j] - o[j])*o[j]*(1-o[j])*x[j][i]

def outputActivationFuncSoftmax(p, j, c, x, i):
    # [DESC]
    # Softmax output activation function
    # p is the probability
    # j is the neuron index
    # c is the target class
    # x is the input
    # i is the input index

    return diffSoftmaxActvFunc(p, j, c) * x[j][i]
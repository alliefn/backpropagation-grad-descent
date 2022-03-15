from numpy import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def linear(x):
    return x


def relu(x):
    return max(0, x)


def softmax(x):
    return exp(x) / exp(x).sum()

from numpy import exp, clip


def sigmoid(x):
    x = clip(x,-500,500)
    return 1 / (1 + exp(-x))


def linear(x):
    return x


def relu(x):
    return max(0, x)


def softmax(x):
    x = clip(x,-500,500)
    return exp(x) / exp(x).sum()

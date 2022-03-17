import numpy as np
import json
from activation_functions import *

# row get from number of neuron in that layer, and col determined from n neuron before this layer


def initRandomBiasWeight(row, col):
    print("Row di fungsi", row)
    print("Col di fungsi", col)
    weights = np.random.randn(row, col)  # generates random 2d array weight
    biases = [0 for _ in range(len(weights))]
    result = weights  # return array 2d with first col is bias and the rest are weights
    result = np.insert(result, 0, biases, axis=1)

    return biases, weights, result


def readFile(filePath: str):
    # Opening JSON file
    f = open(filePath)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Closing file
    f.close()
    return data


def exportOutput(output, filename):  # not with extension
    # Print output to JSON file
    outputData = {"output": output.tolist()}
    with open(filename + ".json", 'w') as outfile:
        json.dump(outputData, outfile)


def calcNet(inputMatrix, hiddenMatrix):
    return np.matmul(inputMatrix, hiddenMatrix.transpose())


def callActivation(category, value):
    if category == "sigmoid":
        return sigmoid(value)
    elif category == "linear":
        return linear(value)
    elif category == "relu":
        return relu(value)
    elif category == "softmax":
        return softmax(value)


def print_hidden_layer( hidden_layer):

    for i in range(len(hidden_layer)):
        print("========================================================")
        print("Hidden Layer :" + str(i))
        print("Activation Function: " +
              str(hidden_layer[i]["activation_function"]))
        print("Unit : " + str(len(hidden_layer[i]["weight"])))
        print("Weight: " + str(hidden_layer[i]["weight"]))
        print("Bias: " + str(hidden_layer[i]["bias"]))
        print("")


def print_output_layer(output_layer):
    print("========================================================")
    print("Output Layer : ")
    print("Activation Function: " +
          str(output_layer["activation_function"]))
    print("Weight: " + str(output_layer["weight"]))
    print("Bias: " + str(output_layer["bias"]))
    print("")


def printModel(modelData):
    print_hidden_layer(modelData["hidden_layer"])
    print_output_layer(modelData["output_layer"])

def calcError(output, target):
    return np.sum(np.square(output - target))

def calcDelta(output, target):
    return output - target

def updateWeight(weight, delta, learning_rate):
    return weight - learning_rate * delta

def updateBias(bias, delta, learning_rate):
    return bias - learning_rate * delta
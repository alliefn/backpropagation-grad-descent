import json
from matplotlib.pyplot import axis
from activation_function import *
import numpy as np


"""
Fungsi untuk membaca file model dan mengisi nilai-nilainya ke variable bersesuaian
"""
def readFile(filePath: str):
    # Opening JSON file
    f = open(filePath)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Closing file
    f.close()
    return data

def exportOutput(output, filename): # not with extension
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
    
def print_hidden_layer(hidden_layer):
    
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

    
    

def predictFeedForward(modelData, inputData):
    hiddenLayers = modelData["hidden_layer"]
    outputLayer = modelData["output_layer"]

    outputLayerActivation = outputLayer["activation_function"]

    # X input as matrix
    for item in inputData["input"]:
        item.insert(0, 1)  # Insert 1 for bias at every input intance

    # Convert to matrix hidden layer
    # adding bias to every neuron h
    for i in range(len(hiddenLayers[0]["bias"])):
        hiddenLayers[0]["weight"][i].insert(0, hiddenLayers[0]["bias"][i])

    # Convert to matrix output layer
    # adding bias to every neuron h
    for i in range(len(outputLayer["bias"])):
        outputLayer["weight"][i].insert(0, outputLayer["bias"][i])

    inputMatrix = np.matrix(inputData["input"])
    hiddenLayerMatrix = np.matrix(hiddenLayers[0]["weight"])
    outputLayerMatrix = np.matrix(outputLayer["weight"])

    for i in range(len(hiddenLayers)):
        hiddenLayerMatrix = np.matrix(
            hiddenLayers[i]["weight"]).astype(float)

        # Start of looping each layer
        hxy = calcNet(inputMatrix, hiddenLayerMatrix)

        # calculate h using activation func
        m, n = hxy.shape
        for row in range(m):
            for col in range(n):
                a = hxy.item(row, col)
                hxy.itemset((row,col),callActivation(
                    hiddenLayers[i]["activation_function"], a))

        # add bias 1
        hxy = np.insert(hxy, 0, [1 for _ in range(len(hxy))], axis=1)
        
        # Forward h value
        # End of loop
        inputMatrix = hxy

    # Calculate to output Layer
    netY = calcNet(hxy, outputLayerMatrix)

    # Compute output using activation function
    for i in range(len(netY)):
        netY[i] = callActivation(outputLayerActivation, netY[i])
    # round all netY values to 5 decimal
    netY = np.round(netY, 5)
    return netY
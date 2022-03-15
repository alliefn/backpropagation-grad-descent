import json
from attr import attributes
import numpy as np
from activation_functions import *


class Backpropagation:
    '''
    n_layer : Jumlah  hidden layer dan output layer [INT]
    array_neuron_layer : array yang berisi jumlah neuron dari tiap layer berukuran n_layer [Array of INT]
    array_activation : kumpulan fungsi aktivasi [Array of STR]
    learning rate : konstanta nilai pembelajaran [NUM]
    error_threshold : batas nilai error [NUM]
    max_iter : jumlah maksimal iterasi [INT]
    batch_size : ukuran batch yang dikerjakan tiap epoch (?) [INT]
    output_per_layer : Nilai result dari setiap layer [Array of array of NUM]
    weight_per_layer : 
    bias_per_layer : 
    '''

    def __init__(self, n_layer, array_neuron_layer, array_activation, learning_rate, error_threshold, max_iter, batch_size):
        self.n_layer = n_layer
        self.array_neuron_layer = array_neuron_layer
        self.array_activation = array_activation
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.output_per_layer = []
        self.weight_per_layer = []
        self.bias_per_layer = []

    '''
    filename : Nama file tempat menyimpan model
    '''

    def load(self, filename):
        # Opening JSON file
        f = open(filename)
        # returns JSON object as a dictionary
        modelData = json.load(f)
        # Closing file
        f.close()

        hiddenLayers = modelData["hidden_layer"]
        outputLayer = modelData["output_layer"]

        # Convert to matrix hidden layer
        # adding bias to every neuron h
        for i in range(len(hiddenLayers)):
            for j in range(len(hiddenLayers[i]["bias"])):
                hiddenLayers[0]["weight"][j].insert(0, hiddenLayers[0]["bias"][j])

        # Convert to matrix output layer
        # adding bias to every neuron h
        for i in range(len(outputLayer["bias"])):
            outputLayer["weight"][i].insert(0, outputLayer["bias"][i])

        hiddenLayers.append(outputLayer)

        self.n_layer = len(hiddenLayers) + 1
        self.neuron_per_layer = [len(layer["weight"])
                                for layer in hiddenLayers]
        self.array_activation = [layer["activation_function"]
                                for layer in hiddenLayers]
        self.weight_per_layer = [layer["weight"][1:] 
                                for layer in hiddenLayers]
        self.bias_per_layer = [layer["weight"][0]
                                for layer in hiddenLayers]
        
        return

    def initRandomBiasWeight(self, row, col): # row get from number of neuron in that layer, and col determined from n neuron before this layer
        weights = np.random.randn(row,col) # generates random 2d array weight
        biases = np.zeros(row)
        result = weights # return array 2d with first col is bias and the rest are weights
        for i in range (row):
           result[i].insert(0, biases[i])

        return biases, weights, result


    def predict(self, input):
        # calculate ffnn
        # abis semuanya dapet attributes
        # calculate backprop
        return

    def predictFeedForward(self, modelData, inputData):
        # hiddenLayers = modelData["hidden_layer"]
        # outputLayer = modelData["output_layer"]

        # outputLayerActivation = outputLayer["activation_function"]

        # X input as matrix
        for item in inputData["input"]:
            item.insert(0, 1)  # Insert 1 for bias at every input intance

        # init random weight and bias for each layer
        weightBiasLayers = []
        # for input layer -> hidden layer
        col = len(inputData["input"][0])
        for i in range(self.n_layer):
            biases, weights, result = self.initRandomBiasWeight(self.array_neuron_layer[i], col) # get first layer input -> hidden
            self.bias_per_layer.append(biases)
            self.weight_per_layer.append(weights)
            weightBiasLayers.append(result)
            col = self.array_neuron_layer[i]


        
        # # Convert to matrix hidden layer
        # # adding bias to every neuron h
        # for i in range(len(hiddenLayers[0]["bias"])):
        #     hiddenLayers[0]["weight"][i].insert(0, hiddenLayers[0]["bias"][i])

        # # Convert to matrix output layer
        # # adding bias to every neuron h
        # for i in range(len(outputLayer["bias"])):
        #     outputLayer["weight"][i].insert(0, outputLayer["bias"][i])

        inputMatrix = np.matrix(inputData["input"])
        hiddenLayerMatrix = np.matrix(hiddenLayers[0]["weight"])
        outputLayerMatrix = np.matrix(outputLayer["weight"])

        for i in range(len(hiddenLayers)):
            hiddenLayerMatrix = np.matrix(
                hiddenLayers[i]["weight"]).astype(float)

            # Start of looping each layer
            hxy = self.calcNet(inputMatrix, hiddenLayerMatrix)

            # calculate h using activation func
            m, n = hxy.shape
            for row in range(m):
                for col in range(n):
                    a = hxy.item(row, col)
                    hxy.itemset((row, col), self.callActivation(
                        hiddenLayers[i]["activation_function"], a))

            # add bias 1
            hxy = np.insert(hxy, 0, [1 for _ in range(len(hxy))], axis=1)

            # Forward h value
            # End of loop
            inputMatrix = hxy

        # Calculate to output Layer
        netY = self.calcNet(hxy, outputLayerMatrix)

        # Compute output using activation function
        for i in range(len(netY)):
            netY[i] = self.callActivation(outputLayerActivation, netY[i])
        # round all netY values to 5 decimal
        netY = np.round(netY, 5)
        return netY

    def readFile(filePath: str):
        # Opening JSON file
        f = open(filePath)

        # returns JSON object as
        # a dictionary
        data = json.load(f)

        # Closing file
        f.close()
        return data

    def exportOutput(self, output, filename):  # not with extension
        # Print output to JSON file
        outputData = {"output": output.tolist()}
        with open(filename + ".json", 'w') as outfile:
            json.dump(outputData, outfile)

    def calcNet(self, inputMatrix, hiddenMatrix):
        return np.matmul(inputMatrix, hiddenMatrix.transpose())

    def callActivation(self, category, value):
        if category == "sigmoid":
            return sigmoid(value)
        elif category == "linear":
            return linear(value)
        elif category == "relu":
            return relu(value)
        elif category == "softmax":
            return softmax(value)

    def print_hidden_layer(self, hidden_layer):

        for i in range(len(hidden_layer)):
            print("========================================================")
            print("Hidden Layer :" + str(i))
            print("Activation Function: " +
                  str(hidden_layer[i]["activation_function"]))
            print("Unit : " + str(len(hidden_layer[i]["weight"])))
            print("Weight: " + str(hidden_layer[i]["weight"]))
            print("Bias: " + str(hidden_layer[i]["bias"]))
            print("")

    def print_output_layer(self, output_layer):
        print("========================================================")
        print("Output Layer : ")
        print("Activation Function: " +
              str(output_layer["activation_function"]))
        print("Weight: " + str(output_layer["weight"]))
        print("Bias: " + str(output_layer["bias"]))
        print("")

    def printModel(self, modelData):
        self.print_hidden_layer(modelData["hidden_layer"])
        self.print_output_layer(modelData["output_layer"])

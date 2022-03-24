import json
import numpy as np
from activation_functions import *
from utils import *


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
        self.weight_bias_layer = []
        self.mse = np.inf

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
                hiddenLayers[0]["weight"][j].insert(
                    0, hiddenLayers[0]["bias"][j])

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

    # def backpropagation(self, inputData, targetData):
    #     # receive input data and target data
    #     # use the backpropagation algorithm to update the weight and bias
    #     # use the forward algorithm to predict the output

    #     # Perform Forward propagation or Forward pass to calculate
    #     # Activation values of neurons in each layer.
    #     netH = self.predictFeedForward(inputData)

    #     # Backpropagation algorithm
    #     # calculate delta
    #     for i in range(self.n_layer - 1, 0, -1):
    #         if i == self.n_layer - 1:
    #             # for output layer
    #             # calculate error
    #             e = calcError(netH, targetData)
    #             # calculate delta
    #             d = calcDelta(self.array_activation[i], netH, targetData)
    #             # # update weight and bias
    #             # self.weight_per_layer[i] = updateWeight(
    #             #     self.weight_per_layer[i], d, self.learning_rate)
    #             # self.bias_per_layer[i] = updateBias(
    #             #     self.bias_per_layer[i], d, self.learning_rate)
    #         else:
    #             # for hidden layer
    #             # calculate error
    #             e = calcError(netH, self.output_per_layer[i + 1])
    #             # calculate delta
    #             d = calcDelta(
    #                 self.array_activation[i], netH, self.output_per_layer[i + 1])
    #             # # update weight
    #             # self.weight_per_layer[i] = updateWeight(
    #             #     self.weight_per_layer[i], d, self.learning_rate)
    #             # self.bias_per_layer[i] = updateBias(
    #             #     self.bias_per_layer[i], d, self.learning_rate)

    # menghitung error term untuk mini batch
    # return error ter
    def calcErrorBatch(self, targetData):
        # start backprop
        layer_err = []
        for i in range(self.n_layer - 1, 0, -1):
            netH = self.output_per_layer[i]
            if i == self.n_layer - 1:
                # for output layer
                # calculate error node
                # e = calcError(netH, targetData)
                for out in range(len(netH)):
                    layer_err.append([])
                    for neuron in range(self.array_neuron_layer[i]):
                        layer_err[out].append(calcErrorOutput(netH[out][neuron], targetData[out][neuron]))
                # calculate delta
                # d = calcDelta(self.array_activation[i], netH, targetData)

            else:
                # for hidden layer
                # calculate error
                new_err = []
                for out in range(len(netH)):
                    new_err.append([])
                    for err in layer_err:
                        new_err[out].append(calcErrorHidden(netH[out][neuron], self.weight_per_layer[i], layer_err[out]))
                
                layer_err = new_err
                # calculate delta
                # d = calcDelta(
                #     self.array_activation[i], netH, self.output_per_layer[i + 1])
        

    def backpropagation(self, inputData, targetData):
        epoch = 0
        error = np.inf
        self.initWeightBiasRandom(inputData)
        
        while(epoch < self.max_iter and (error > self.error_threshold)):
            no_of_batches = int(len(inputData) / self.batch_size) # assumed batch size is factor of inputData

            for j in range(no_of_batches):

                # Initialize delta
                delta = []
                deltaBias = []
                for row in (j*self.batch_size, (j+1)*self.batch_size):

                    # instance["input"] = inputData["input"][row] --> [x1 ,x2 , x3]
                    # Feed forward mini batch
                    minibatchInput = inputData["input"][j*self.batch_size : (j+1)*self.batch_size]
                    netH = self.predictFeedForward(minibatchInput) 
                   
                    # start backprop

                    # calculate error term

                    layer_err = []
                    for i in range(self.n_layer - 1, 0, -1):
                        if i == self.n_layer - 1:
                            # for output layer
                            # calculate error node
                            # e = calcError(netH, targetData)
                            for out in range(len(netH)):
                                for neuron in range(self.array_neuron_layer[i]):
                                    layer_err.append(calcErrorOutput(netH[out][neuron], targetData[out][neuron]))
                            # calculate delta
                            # d = calcDelta(self.array_activation[i], netH, targetData)

                        else:
                            # for hidden layer
                            # calculate error
                            for err in layer_err:
                                e = calcErrorHidden(netH, self.weight_per_layer[i], layer_err)
                            # calculate delta
                            # d = calcDelta(
                            #     self.array_activation[i], netH, self.output_per_layer[i + 1])

                    # Calculate delta
                    delta.append(calcDelta(
                        self.array_activation[self.n_layer - 1], netH, targetData[row]))
                        
                    # Calculate delta bias
                    deltaBias.append(calcDelta(
                        self.array_activation[self.n_layer - 1], netH, targetData[row]))
        
                # Update weight and bias with its delta value, learning rate and batch size
                # delta and deltaBias is an array, so we need to iterate through it
                for i in range(len(delta)):

                    self.weight_per_layer[i] = updateWeight(
                        self.weight_per_layer[i], delta[i], self.learning_rate, self.batch_size)

                    self.bias_per_layer[i] = updateBias(
                        self.bias_per_layer[i], deltaBias[i], self.learning_rate, self.batch_size)

            # ErrorMSE =      
            epoch += 1  

    def predict(self, inputData):
        # receive new observation (x_i .. x_n) and return the prediction y
        # use the forward alnorithm to predict the output
        self.initWeightBiasRandom(inputData)
        # forward algorithm
        netH = self.predictFeedForward(inputData)
        # starts the backpropagation algorithm
        # backpropagation algorithm
        # while threshold not reached, continue
        epoch = 0
        while (self.mse > self.error_threshold and epoch < self.max_iter ):
            
            # Mini batch backpropagation
            for i in range(0, len(inputData["input"]), self.batch_size):
                # get the mini batch
                mini_batch = inputData["input"][i:i+self.batch_size]
                # forward algorithm
                netH = self.predictFeedForward(mini_batch)
                # backpropagation algorithm
                self.backpropagation(mini_batch, netH)
            epoch += 1

    def initWeightBiasRandom(self, inputData):
        col = len(inputData["input"][0])
        # for input layer -> hidden layer
        for i in range(self.n_layer):
            biases, weights, weight_bias = initRandomBiasWeight(
                self.array_neuron_layer[i], col)
            self.bias_per_layer.append(biases)
            self.weight_per_layer.append(weights)
            self.weight_bias_layer.append(weight_bias)
            # jumlah neuron dari layer sebelumnya
            col = self.array_neuron_layer[i]

    # input Data =  [ [x1, x2, x3, x4], [x1, x2, x3, x4], [x1, x2, x3, x4] ]
    # netH = 2d, setiap baris menandakan data ke-i
    def predictFeedForward(self, inputData):
        # reset output per layer
        self.output_per_layer = []
        # X input as matrix
        for item in inputData: # before = inputData["input"]
            item.insert(0, 1)  # Insert 1 for bias at every input intance
        inputMatrix = np.matrix(inputData) # before = inputData["input"]

        # for input layer -> hidden layer
        for i in range(self.n_layer):
            neuronLayerMatrix = np.matrix(self.weight_bias_layer[i]).astype(float)

            # Start of looping each layer
            netH = calcNet(inputMatrix, neuronLayerMatrix)
            # calculate h using activation func
            m, n = netH.shape
            for r in range(m):
                for c in range(n):
                    a = netH.item(r, c)
                    netH.itemset((r, c), callActivation(
                        self.array_activation[i], a))

            # save output per layer
            self.output_per_layer.append(netH)
            # add bias 1
            inputMatrix = np.insert(
                netH, 0, [1 for _ in range(len(netH))], axis=1)

            # Forward h value
            # End of loop
        netH = np.round(netH, 5)
        return netH

    def printInfo(self):
        print("N Layer : ", self.n_layer)
        print("Array neuron layer : ")
        print(self.array_neuron_layer)
        print("Array activation : ")
        print(self.array_activation)
        print("Learning rate : ", self.learning_rate)
        print("Error threshold ", self.error_threshold)
        print("Max iter ", self.max_iter)
        print("Batch size ", self.batch_size)
        print("Output per layer")
        print(self.output_per_layer)
        print("weight per layer ")
        print(self.weight_per_layer)
        print("Bias per layer ")
        print(self.bias_per_layer)

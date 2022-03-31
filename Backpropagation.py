import json
import numpy as np
from activation_functions import *
from utils import *
import copy


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
    bias_per_layer : Weight bias dari tiap neuron
    
    '''

    def __init__(self, n_layer, array_neuron_layer, array_activation, learning_rate, error_threshold, max_iter, batch_size):
        self.n_layer = n_layer
        self.array_neuron_layer = array_neuron_layer
        self.array_activation = array_activation
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.net_per_layer = []
        self.output_per_layer = []
        self.weight_per_layer = []
        self.bias_per_layer = []
        self.weight_bias_layer = []
        self.error_term = []

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

        print(hiddenLayers)

        self.n_layer = len(hiddenLayers)
        self.array_neuron_layer = [len(layer["weight"])
                                 for layer in hiddenLayers]
        self.array_activation = [layer["activation_function"]
                                 for layer in hiddenLayers]
        matrix = []
        for layer in hiddenLayers:
            layerweight = []
            for neuron in layer["weight"]:
                layerweight.append(neuron[1:])
            matrix.append(layerweight)

        self.weight_per_layer = np.ndarray(matrix)

        self.bias_per_layer = [layer["bias"]
                               for layer in hiddenLayers]

        return

    """
    Menghitung error term untuk mini batch
    @return array 3 dimensi, dimensi pertama menandakan error term tiap layer untuk seluruh instance, dalam setiap layer baris menandakan instance ke-x, dan kolom menandakan neuron ke-y
    """
    def calculateErrorTerm(self, y_true : List):
        """
        [DESC]
        Menghitung error term untuk mini batch
        @return structure 
        Array of per layer -> array instance -> array neuron
        """
        self.error_term = []
        # start backprop
        layer_err = []
        for i in range(self.n_layer - 1, -1, -1):
            y_pred = np.asarray(self.output_per_layer[i])
            netH = np.asarray(self.net_per_layer[i])
            if i == self.n_layer - 1:
                new_err = []
                for out in range(len(netH)):
                    # layer_err.append([])
                    error_per_neuron = []
                    for neuron in range(self.array_neuron_layer[i]):
                        if (self.array_activation[i] == "softmax"):
                            j = neuron
                            c = y_true[out].index(1)
                            # c = np.where(y_true[out] == 1)[0][0]
                            p = y_pred[out][0]
                            error_term = calcErrorOutputSoftmax(p,j,c)
                        else:
                            error_term = calcErrorOutput(netH[out][neuron], y_true[out][neuron], self.array_activation[i])
                        # layer_err[out].append(error_term)
                        error_per_neuron.append(error_term)
                    new_err.append(error_per_neuron)
                self.error_term.insert(0, new_err)
                layer_err = new_err
            else:
                # for hidden layer
                new_err = []
                for out in range(len(netH)): # untuk setiap instance
                    error_per_neuron = []
                    for neuron in range(self.array_neuron_layer[i]):
                        # get neuron weight
                        neuron_weight = []
                        weightPerLayer = self.weight_per_layer[i+1]
                        # separate value weight for each neuron
                        for idx_weight in range (len(weightPerLayer)):
                            neuron_weight.append(weightPerLayer[idx_weight][neuron])
                        if (self.array_activation[i] == "softmax"):
                            j = neuron
                            c = y_true[out].index(1)
                            # c = np.where(y_true[out] == 1)[0][0]
                            p = y_pred[out][0] # output neuron
                            error_term = calcErrorHiddenSoftmax( neuron_weight, layer_err[out], p,j,c)
                        else:
                            error_term = calcErrorHidden(netH[out][neuron], neuron_weight, layer_err[out], self.array_activation[i])
                        # print("Error term hidden", error_term)
                        error_per_neuron.append(error_term)
                    new_err.append(error_per_neuron)
                self.error_term.insert(0, new_err)
                # update layer error term di next layer
                layer_err = new_err
        
    """
    Fungsi untuk melatih model
    """
    def backpropagation(self, X_train : List, y_train : List):
        # targetData = y asli yang sudah di encode
        inputData = X_train
        targetData = y_train
        epoch = 0
        error = np.inf
        self.initWeightBiasRandom(inputData)
        while(epoch < self.max_iter and (error > self.error_threshold)):
            
            no_of_batches = int(len(inputData) / self.batch_size) # assumed batch size is factor of inputData
            error = 0
            for j in range(no_of_batches):
                # instance["input"] = inputData["input"][row] --> [x1 ,x2 , x3]
                # Feed forward mini batch
                minibatchInput = copy.deepcopy(inputData[j*self.batch_size : (j+1)*self.batch_size])
                output_h = self.predictFeedForward(minibatchInput) 
                # start backprop
                # calculate error term
                self.calculateErrorTerm(targetData)

                self.updateWeight()
                
                for instance in range(len(output_h)):
                    for neuron in range(self.array_neuron_layer[-1]):
                        error += pow(targetData[instance][neuron] - output_h[instance][neuron], 2)
                        # print("selisih ", targetData[instance][neuron] - output_h[instance][neuron])
            
            error = error / 2
            print("---------------------")
            print("At Epoch", epoch ,"Error : ", error)
            # self.debug()
            print("---------------------")
            epoch += 1  

    def predict(self, X_test : List):
        # ASUMSI : dipanggil setelah fit
        # receive new observation (x_i .. x_n) and return the prediction y
        # use the forward alnorithm to predict the output
        result = self.predictFeedForward(X_test)
        print(result)
        predictedValue = result.tolist()
    
        # encode predicted result
        return [ predictedValue[x].index(max(predictedValue[x])) for x in range(len(X_test))]

    def initWeightBiasRandom(self, inputData):
        col = len(inputData[0])
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
        self.net_per_layer = []
        # X input as matrix
        for item in inputData: # before = inputData["input"]
            item.insert(0, 1)  # Insert 1 for bias at every input intance

        inputMatrix = np.matrix(inputData) # before = inputData["input"])))
        # for input layer -> hidden layer
        for i in range(self.n_layer):
            neuronLayerMatrix = np.matrix(self.weight_bias_layer[i]).astype(float)
            # Start of looping each layer
            netH = calcNet(inputMatrix, neuronLayerMatrix)
            self.net_per_layer.append(netH)
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
        netH = np.round(netH, 5) # output 
        return netH

    def updateWeight(self):
        # Fungsi update weight di akhir batch
        
        sum_delta = [None for _ in range(len(self.weight_per_layer))]
        sum_delta_bias = [None for _ in range(len(self.bias_per_layer))]
# 
        for idx_layer,layer in enumerate(self.error_term): # for every layer do update weight

            sum_delta[idx_layer] = [None for _ in range(len(self.weight_per_layer[idx_layer]))]
            sum_delta_bias[idx_layer] = [0 for _ in range(len(self.bias_per_layer[idx_layer]))]

            for instance in layer: #for each instance get all delta
                
                for i in range(len(instance)):

                    sum_delta[idx_layer][i] = [0 for _ in range(len(self.weight_per_layer[idx_layer][i]))]

                    for j in range(len(self.weight_per_layer[idx_layer][i])):
                        sum_delta[idx_layer][i][j] += np.round(instance[i]*self.learning_rate*self.output_per_layer[idx_layer].item(j),5)
                        
                    sum_delta_bias[idx_layer][i] += np.round(instance[i]*self.learning_rate*1,5)
                
            for i in range(len(self.bias_per_layer[idx_layer])): #add sum delta to update weight
                self.bias_per_layer[idx_layer][i] += sum_delta_bias[idx_layer][i]
                self.weight_bias_layer[idx_layer][i][0] += sum_delta_bias[idx_layer][i]
            
                    
            for i in range(len(self.weight_bias_layer[idx_layer])): #add sum delta to update weight
                for j in range(len(self.weight_bias_layer[idx_layer][i])):

                    self.weight_per_layer[idx_layer][i] += sum_delta[idx_layer][i]

                    if j != 0:
                        
                        self.weight_bias_layer[idx_layer][i][j] += sum_delta[idx_layer][i][j-1]

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

    def print_hidden_layer(self):
        for i in range(self.n_layer-1):
            print("========================================================")
            print("Hidden Layer-" + str(i+1) + " :")
            print("Activation Function: " +
                str(self.array_activation[i]))
            print("Unit : " + str(self.array_neuron_layer[i]))
            print("Weight: " + str(self.weight_per_layer[i]))
            print("Weight Bias: " + str(self.bias_per_layer[i]))
            print("")
            
    def print_output_layer(self):
        i = self.n_layer-1
        print("========================================================")
        print("Output Layer : ")
        print("Activation Function: " +
            str(self.array_activation[i]))
        print("Unit : " + str(self.array_neuron_layer[i]))
        print("Weight: " + str(self.weight_per_layer[i]))
        print("Weight Bias: " + str(self.bias_per_layer[i]))
        print("")

    def print_layer(self, i) :
        print("========================================================")
        print("Hidden Layer-" + str(i+1) + " :")
        print("Activation Function: " +
            str(self.array_activation[i]))
        print("Unit : " + str(self.array_neuron_layer[i]))
        print("Weight: " + str(self.weight_per_layer[i]))
        print("Weight Bias: " + str(self.bias_per_layer[i]))
        print("")

    def printModel(self):
        self.print_hidden_layer()
        self.print_output_layer()

    def debug(self):
        print("Error Term: ", self.error_term)
        self.printModel()
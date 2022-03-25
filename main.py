from utils import *
from Backpropagation import *

def main():
    fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = processCSV(fileInput)

    backprop = Backpropagation(n_layer = 2, array_neuron_layer=[2,1], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)
    backprop.initWeightBiasRandom(inputData)
    print(backprop.weight_bias_layer)
    print("============================")
    print(backprop.weight_per_layer)
    print("============================")
    print(backprop.bias_per_layer)
    # ffnn = backprop.predictFeedForward(inputData=inputData)
    print("memasuki backp")
    # backprop.backpropagation(inputData=inputData, targetData=inputData)

    print("Info")
    # backprop.printInfo()
    print("-------------------------")
        

if __name__ == "__main__":
    main()
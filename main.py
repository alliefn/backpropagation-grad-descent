from utils import *
from Backpropagation import *

def main():
    fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = processCSV(fileInput)
    print(inputData['input'])
    print("===============================")
    print(inputData['output'])
    backprop = Backpropagation(n_layer = 3, array_neuron_layer=[2,3,3], array_activation=["sigmoid", "relu", 'sigmoid'], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)
    backprop.initWeightBiasRandom(inputData['input'])
    print("============================")
    # print(backprop.weight_bias_layer)
    print("Weight bias")
    for i in backprop.weight_bias_layer:
        print(i)
        print("====")
    print("===============================")
    ffnn = backprop.predictFeedForward(inputData=inputData['input'])
    print("============================")
    print(ffnn)
    # backprop.backpropagation(inputData=inputData, targetData=inputData['output'])

    print("Info")
    # backprop.printInfo()
    print("-------------------------")
        

if __name__ == "__main__":
    main()
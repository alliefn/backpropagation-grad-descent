from utils import *
from Backpropagation import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

def main():
    # fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    # inputData = processCSV(fileInput)
    inputData = load_iris()
    encoder = OneHotEncoder(sparse=False)
    reshape = inputData["target"].reshape(len(inputData["target"]), 1)
    target = encoder.fit_transform(reshape)
    
    # print(inputData['input'])
    # print("===============================")
    # print(inputData['output'])
    backprop = Backpropagation(n_layer = 3, array_neuron_layer=[2,3,3], array_activation=["sigmoid", "relu", 'sigmoid'], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)
    # backprop.initWeightBiasRandom(inputData['input'])
    # print("============================")
    # # print(backprop.weight_bias_layer)
    # print("Weight bias")
    # for i in backprop.weight_bias_layer:
    #     print(i)
    #     print("====")
    # print("===============================")
    # ffnn = backprop.predictFeedForward(inputData=inputData['input'])
    # print("============================")
    # print(ffnn)
    backprop.backpropagation(inputData, target)

    print("Info")
    backprop.printInfo()
    print("-------------------------")
        

if __name__ == "__main__":
    main()
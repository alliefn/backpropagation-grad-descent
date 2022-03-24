from utils import *
from Backpropagation import *

def main():
    fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = readFile(fileInput)

    print(inputData["input"][0:2])
    # backprop = Backpropagation(n_layer = 2, array_neuron_layer=[2,1], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)
    
    # ffnn = backprop.predictFeedForward(inputData=inputData)
    print("memasuki backp")
    # backprop.backpropagation(inputData=inputData, targetData=inputData)

    print("Info")
    # backprop.printInfo()
    print("-------------------------")
        

if __name__ == "__main__":
    main()
from utils import *
from Backpropagation import *

def main():
    fileModel = input("Masukan file model : ")
    fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = readFile(fileInput)

    backprop = Backpropagation(n_layer = 2, array_neuron_layer=[2,1], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)

    ffnn = backprop.predictFeedForward(inputData=inputData)

    print("Info")
    backprop.printInfo()
    print("-------------------------")

    # Print model here
    print(ffnn)
        

if __name__ == "__main__":
    main()
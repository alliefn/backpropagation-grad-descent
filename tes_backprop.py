from utils import *
from Backpropagation import *

def main():
    # fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = readFile("model/input.json")
    y_true = [[1,0], [0,1], [0,1], [1,0]]
    backprop = Backpropagation(n_layer = 2, array_neuron_layer=[3,2], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=4)
    backprop.backpropagation(inputData['input'], y_true)
    # backprop.predictFeedForward(inputData=inputData['input'])
    # backprop.calculateErrorTerm(y_true)
    predicted = backprop.predict(inputData['input'])
    print("tipe inputData", type(inputData['input']))
    print("tipe target", type(y_true))
    print("Predicted Value")
    print(predicted)
    print()
    # print score accuracy
    print("Score Accuracy")
    print(score_accuracy(predicted, [1,1,1,1]))
    print()
    print("Info")
    backprop.printModel()
    print("-------------------------")
if __name__ == "__main__":
    main()
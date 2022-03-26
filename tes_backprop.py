from utils import *
from Backpropagation import *

def main():
    # fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = readFile("model/input.json")
    y_true = np.array([[1,0], [0,1], [0,1], [1,0]], np.float32)
    backprop = Backpropagation(n_layer = 2, array_neuron_layer=[3,2], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=4)
    backprop.backpropagation(inputData['input'], y_true)
    # backprop.predictFeedForward(inputData=inputData['input'])
    # backprop.calculateErrorTerm(y_true)
    
if __name__ == "__main__":
    main()
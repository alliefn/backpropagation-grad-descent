from utils import *
from Backpropagation import *

def main():
    fileInput = input("Masukan file input : ")

    # modelData = readFile(fileModel)
    inputData = readFile(fileInput)
    y_true = np.array([[1], [0], [0], [1]], np.float32)
    backprop = Backpropagation(n_layer = 2, array_neuron_layer=[2,1], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)
    backprop.initWeightBiasRandom(inputData)
    backprop.predictFeedForward(inputData=inputData['input'])
    backprop.calculateErrorTerm(y_true)

    print("Weight bias layer")
    print(backprop.weight_bias_layer)
    
if __name__ == "__main__":
    main()
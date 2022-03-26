from utils import *
from Backpropagation import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

def main():
    # load iris data
    inputData = load_iris()
    encoder = OneHotEncoder(sparse=False)
    reshape = inputData["target"].reshape(len(inputData["target"]), 1)
    target = encoder.fit_transform(reshape)
    
    backprop = Backpropagation(n_layer = 10, array_neuron_layer=[6,6,6,6,6,6,6,6,6,3], array_activation=["sigmoid", "relu", 'sigmoid', "sigmoid","sigmoid","sigmoid", "softmax", "linear", "sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)

    backprop.backpropagation(inputData, target)

    print("Info")
    backprop.printInfo()
    print("-------------------------")
        

if __name__ == "__main__":
    main()
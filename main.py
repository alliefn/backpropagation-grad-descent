from utils import *
from Backpropagation import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

def main():
    # load iris data
    inputData = load_iris()
    target_unencoded = inputData.target
    encoder = OneHotEncoder(sparse=False)
    reshape = inputData["target"].reshape(len(inputData["target"]), 1)
    target = encoder.fit_transform(reshape)

    # Define parameters
    n_layer = 10
    array_neuron_layer = [6,6,6,6,6,6,6,6,6,3]
    array_activation = ["sigmoid", "sigmoid", 'sigmoid', "sigmoid","sigmoid","sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"]
    learning_rate = 0.001
    error_threshold = 1
    max_iter = 300
    batch_size = 30
    # create model
    backprop = Backpropagation(n_layer = n_layer, array_neuron_layer=array_neuron_layer, array_activation=array_activation, learning_rate=learning_rate, error_threshold=error_threshold, max_iter=max_iter, batch_size=batch_size)

    # train model
    inputData = inputData["data"].tolist()
    target = target.tolist()
    print("tipe inputData", type(inputData))
    print("tipe target", type(target))
    backprop.backpropagation(inputData, target)
    # backprop.initWeightBiasRandom(inputData)
    predicted = backprop.predict(inputData)
    print("Predicted Value")
    print(predicted)
    print()
    # print score accuracy
    print("Score Accuracy")
    print(score_accuracy(predicted, target_unencoded))
    print()
    print("Info")
    backprop.printModel()
    print("-------------------------")
        

if __name__ == "__main__":
    main()